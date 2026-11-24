#pragma once

//
// T: float/double (real symmetric) or gpu_complex<> (complex Hermitian); d_ and
// e_ are always real.  Hermitian case: same reflector convention as
// tred1_tiled_ (H = I + beta*u*u^H, e real); the delayed rank-2 updates are
// A += U V^H + V U^H.  For real T conj/real/norm are identities.
//
template <class T>
__device__ __noinline__ void
//__device__ __forceinline__ void
hhsy2tr_(const int nm, const int n, T * __restrict__ a_, real_t<T> * __restrict__ d_, real_t<T> * __restrict__ e_, const int mb, T * __restrict__ u_, T * __restrict__ v_)
{
  typedef real_t<T> TR;
  sync_on_warp();
  const int myid = threadIdx.x % WARP_GPU_SIZE + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define u(row,col)      (*(u_+((row)-1)+((col)-1)*nm))
#define v(row,col)      (*(v_+((row)-1)+((col)-1)*nm))
#define d(index)        (*(d_+((index)-1)))
#define e(index)        (*(e_+((index)-1)))
#define um(index)       (*(um_+((index)-1)))
#define vm(index)       (*(vm_+((index)-1)))
#define uk(index)       (*(uk_+((index)-1)))
#define vk(index)       (*(vk_+((index)-1)))

  const int tile_size = WARP_GPU_SIZE;
  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>( 0.0e0);
  const T ONE  = static_cast<T>( 1.0e0);


  _if_ (n==1) { return; }
  _if_ (!scalar_traits<T>::is_complex && n==2) {
    _if_ (myid==1) {
      const TR t = Abs(a(1,2));
      d(1) = real(a(1,1));
      d(2) = real(a(2,2));
      a(1,1) = ZERO;
      a(2,2) = -t;
      a(1,2) = 2*t;
      e(1) = -t;
      e(2) = static_cast<TR>(0);
    } sync_on_warp();
    return;
  }

  // the diagonal of a Hermitian matrix is real
  if (scalar_traits<T>::is_complex) {
    for(int j=myid; j<=n; j+=WARP_GPU_SIZE) { zero_imag(a(j,j)); }
    sync_on_warp();
  }


#if defined(__NVCC__)
  const int XDIM = 16; //(std::is_same<T,double>::value) ? 8 : 16;
#else
  const int XDIM = 8;
#endif
  const int YDIM = WARP_GPU_SIZE/XDIM;
  const int xid = ((myid-1) % XDIM)+1;
  const int yid = ((myid-1) / XDIM)+1;


  const int ib0 = (n-1)/mb+1;
  const int ib1 = max(1,2-mb);

#pragma unroll 1
  for(int ib=ib0; ib>=ib1; ib--) {

    const int i0 = (ib-1)*mb;
    const int i1 = min(i0+mb,n);
    const int m0 = i1 - i0;
    const int m1 = max(1,2*(2-ib));

#pragma unroll 1
    for(int m=m0; m>=m1; m--) {
      const int i = i0 + m;
      const int l = i - 1;

      T * um_ = (&u(1,m));
      T * vm_ = (&v(1,m));

      for(int j=myid; j<=i1; j+=WARP_GPU_SIZE) {
        um(j) = vm(j) = ZERO;
      }


      T beta; {
        T * ui_ = shmem;
        T * vi_ = shmem + 16;
	{ int k=m+myid; _if_ ( k<=m0 ) {
          ui_[k-1] = u(i,k);
          vi_[k-1] = v(i,k);
	}} sync_on_warp();
        TR anorm = static_cast<TR>(0);
        for(int j=myid; j<=i; j+=WARP_GPU_SIZE) {
          T uj = um(j) = a(j,i);
          for(int k=m0; k>=m+1; k--) {
            uj += u(j,k) * conj(vi_[k-1]);
            uj += v(j,k) * conj(ui_[k-1]);
	  }
          um(j) = uj;
          uj = __MASK__(uj, j<=l);
          anorm = anorm + norm(uj);
        } sum_on_warp(anorm);
        _if_ ( myid == 1 ) {
          d(i) = real(um(i));
          um(i) = ZERO;
          T ul = um(l);
          e(l) = anorm = - Sign(Sqrt(anorm), real(ul));
          um(l) = ul = ul - static_cast<T>(anorm);
          //beta = __MASK__(ONE, anorm != ZERO) / flip0to1(ul * anorm);
          beta = Reciprocal(flip0to1(conj(ul) * static_cast<T>(anorm)));
        } bcast_on_warp(beta,1);
      }


      // v := v - (UV + VU) * u
      _if_ (m0>=m+1){
        T f = ZERO;
        T g = ZERO;
        {
          T *uk_ = &u(myid,m0);
          T *vk_ = &v(myid,m0);
          T *uu_ = &um(myid);
          for(int j=myid; j<=l; j+=WARP_GPU_SIZE) {
            const T h = *uu_;
            f = f + conj(*vk_) * h;
            g = g + conj(*uk_) * h;
            uk_ += WARP_GPU_SIZE; vk_ += WARP_GPU_SIZE; uu_ += WARP_GPU_SIZE;
          }
        } sum2_on_warp(f,g);
#pragma unroll 1
        for(int k=m0-1; k>=m+1; k--) {
          const T ff = f;
          const T gg = g;
          f = ZERO;
          g = ZERO;
          T *uk_ = &u(myid,k);
          T *vk_ = &v(myid,k);
          T *uu_ = &um(myid);
          T *vv_ = &vm(myid);
          for(int j=myid; j<=l; j+=WARP_GPU_SIZE) {
            const T h = *uu_;
            T z = *vv_;
            f = f + conj(*vk_) * h;
            g = g + conj(*uk_) * h;
            z = z + ff * (*(uk_+nm)) + gg * (*(vk_+nm));
            *vv_ = z;
            uk_ += WARP_GPU_SIZE; vk_ += WARP_GPU_SIZE; uu_ += WARP_GPU_SIZE; vv_ += WARP_GPU_SIZE;
          } sum2_on_warp(f,g);
        }
        {
          T *uk_ = &u(myid,m+1);
          T *vk_ = &v(myid,m+1);
          T *vv_ = &vm(myid);
          for(int j=myid; j<=l; j+=WARP_GPU_SIZE) {
            T z = *vv_;
            z = z + f * (*uk_) + g * (*vk_);
            *vv_ = z;
            uk_ += WARP_GPU_SIZE; vk_ += WARP_GPU_SIZE; vv_ += WARP_GPU_SIZE;
          }
        }
      }


      // v := A * u
      const int lx1 = (l & 0x1);
      _if_ (lx1){ T vj = static_cast<T>(real(a(1,1))) * um(1); _if_(myid==lx1) { vm(1) += vj; }}
#pragma unroll 1
      for(int k=1+lx1; k<=l; k+=2) {
       	sync_on_warp();
        T * aj0 = &a(myid,k+0);
        const T uk0 = um(k+0);
        const T uk1 = um(k+1);
        T vk0 = ZERO;
        T vk1 = ZERO;
        for(int j=myid; j<=k+1; j+=WARP_GPU_SIZE,aj0+=WARP_GPU_SIZE) {
          const int km = j-k;
          const T ajk0 = *(aj0);
          const T ajk1 = *(aj0+nm);
          const T uj = um(j);
          const T vj = vm(j);
          T vjj = vj + ajk0 * uk0 + ajk1 * uk1;
          _if_ (km<=0) { vm(j) = vjj; }
          // v(k) += conj(a(j,k)) * u(j)   (a(k,j) = conj(a(j,k)));
          // masked after the product so that the unreferenced lower triangle
          // never enters an arithmetic expression
          vk0 += __MASK__(conj(ajk0) * uj, km<=-1);
          vk1 += conj(ajk1) * uj;
        } T const vkk = red2_on_warp(vk0,vk1,myid-1);
        _if_ (myid<=2) { vm(k+myid-1) += vkk; }
      }


      // v := v + alpha*u
      T alpha = ZERO;
      for(int j=myid; j<=l; j+=WARP_GPU_SIZE) {
        const T f = vm(j) * beta;
        alpha = alpha + conj(f) * um(j);
        vm(j) = f;
      } sum_on_warp(alpha);
      alpha *= (beta * static_cast<T>(0.5));
      for(int j=myid; j<=l; j+=WARP_GPU_SIZE) {
        vm(j) = vm(j) + alpha * um(j);
      } sync_on_warp();
    }

    
    // A := A + UV + VU
    {
    const int i2 = i0 + m1 - 1;
    const int mm = m0 - m1 + 1;
    const int BLK_J = 3;
//    const int BLK_J = 4;
    const int BLK_K = 3;
//    const int BLK_K = 4;
//    const int BLK_M = 2;
//    const int BLK_M = 3;
    const int BLK_M = 4;
//    const int BLK_M = 8;
#pragma unroll 1
    for(int k0=1; k0<=i2; k0+=YDIM*BLK_K) { int k=k0+BLK_K*(yid-1);
#pragma unroll 1
    for(int j0=1; j0<=i2; j0+=XDIM*BLK_J) { int j=j0+BLK_J*(xid-1);
    _if_ (j0+1<=k0+YDIM*BLK_K) {

//      T ajk[BLK_J][BLK_K];
      T ajk[BLK_K][BLK_J];
      for(int K=0;K<BLK_K;K++){
      for(int J=0;J<BLK_J;J++){
        ajk[K][J] = ZERO;
      }}
      for(int K=0;K<BLK_K;K++){
      _if_(k+K<=i2) {
      for(int J=0;J<BLK_J;J++){
      _if_(j+J<=k+K) {
        ajk[K][J]= a(j+J,k+K);
      }}}} 

      T *ujm = &u(j,m1);
      T *vjm = &v(j,m1);
      T *ukm = &u(k,m1);
      T *vkm = &v(k,m1);

//      T uj[BLK_J][BLK_M], uk[BLK_K][BLK_M], vj[BLK_J][BLK_M], vk[BLK_K][BLK_M];
      T uj[BLK_M][BLK_J], uk[BLK_M][BLK_K], vj[BLK_M][BLK_J], vk[BLK_M][BLK_K];

#if defined(__NVCC__)
      asm volatile ("//--open");
#endif
#pragma unroll 1
      for(int m=m1; m<=m1+(mm%BLK_M)-1; m++) {
          {int M=0;
          for(int J=0;J<BLK_J;J++){
            uj[M][J] = ujm[J];
            vj[M][J] = vjm[J];
          } ujm+=nm; vjm+=nm;
          for(int K=0;K<BLK_K;K++){
            uk[M][K] = ukm[K];
            vk[M][K] = vkm[K];
	  } ukm+=nm; vkm+=nm;
          }
          {int M=0;
          for(int K=0;K<BLK_K;K++){
          for(int J=0;J<BLK_J;J++){
            ajk[K][J] = ajk[K][J]
                      + uj[M][J] * conj(vk[M][K]) + vj[M][J] * conj(uk[M][K]);
          }}}
      }

#if defined(__NVCC__)
      asm volatile ("//--mid");
#endif

#pragma unroll 1
      for(int m=m1+(mm%BLK_M); m<=m0; m+=BLK_M) {
          for(int M=0;M<BLK_M;M++){
          for(int J=0;J<BLK_J;J++){
            uj[M][J] = ujm[J];
            vj[M][J] = vjm[J];
	  } ujm+=nm; vjm+=nm; }
          for(int M=0;M<BLK_M;M++){
          for(int K=0;K<BLK_K;K++){
            uk[M][K] = ukm[K];
            vk[M][K] = vkm[K];
	  } ukm+=nm; vkm+=nm; }
          for(int M=0;M<BLK_M;M++){
          for(int K=0;K<BLK_K;K++){
          for(int J=0;J<BLK_J;J++){
            ajk[K][J] = ajk[K][J]
                      + uj[M][J] * conj(vk[M][K]) + vj[M][J] * conj(uk[M][K]);
          }}}
      }
#if defined(__NVCC__)
      asm volatile ("//--close");
#endif

      for(int K=0;K<BLK_K;K++){
      _if_(k+K<=i2) {
      for(int J=0;J<BLK_J;J++){
      _if_(j+J<=k+K) {
        T x = ajk[K][J];
        _if_(j+J==k+K) { zero_imag(x); }
        a(j+J,k+K) = x;
      }}}}
    }}}}
    sync_on_warp();

#pragma unroll 1
    for(int m=m0; m>=m1; m--) {
      const int i = i0 + m;
      for(int j=myid; j<=i; j+=WARP_GPU_SIZE) {
        a(j,i) = u(j,m);
      }
    }

  }
  sync_on_warp();

  _if_ (myid==1) {
    d(1) = real(a(1,1));
    e(n) = static_cast<TR>(0);
  }
  sync_on_warp();

  for(int j=myid+1; j<=n; j+=WARP_GPU_SIZE) {
    a(j,j) = static_cast<T>(e(j-1));
  }


#undef a
#undef u
#undef v
#undef d
#undef e
#undef um
#undef vm
#undef uk
#undef vk
  sync_on_warp();
}

