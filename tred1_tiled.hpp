#pragma once

//
// T is the matrix scalar type: float/double (real symmetric) or
// gpu_complex<float/double> (complex Hermitian).  TR is the real type.
//
// Hermitian case: A := H^H A H with H = I + beta*u*u^H (complex beta, unitary,
// not Hermitian), chosen so that the sub/super-diagonal e is REAL
// (as zhetd2).  u is kept in a(1:l,i), e(i-1) in a(i,i), and beta is
// recoverable as 1/(e(i-1)*conj(u(l))).  Q = H(n)...H(2).
// For real T, conj/real/norm are identities and this is the original code.
//
template <class T, int tile_size>
__device__  __noinline__ void
//__device__  __forceinline__ void
tred1_tiled_(const int nm, const int n, T * __restrict__ a_)
{
  sync_on_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define        a(row,col)        (*(a_+((row)-1)+((col)-1)*nm))
#define        u(index)        (*(u_+((index)-1)))
#define        v(index)        (*(v_+((index)-1)))
#define        d(index)        (*(d_+((index)-1)))
#define        e(index)        (*(e_+((index)-1)))

  typedef real_t<T> TR;
  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);
  const T MONE = static_cast<T>(-1.0e0);

  // d and e are real; they fit in the first tile_size elements of T.
  TR * const d_ = (TR *)shmem;
  TR * const e_ = d_ + tile_size;

  _if_ (n<=1) { return; }
  // The real case needs no reflector for n==2.  The Hermitian case does
  // (it makes e real), so it goes through the general loop below.
  _if_ (!scalar_traits<T>::is_complex && n==2) {
    _if_ (myid==1) {
      d(1) = real(a(1,1));
      d(2) = real(a(2,2));
      e(1) = real(a(1,2));
      e(2) = static_cast<TR>(0);
    } sync_on_cg<T,tile_size>();
    return;
  }

#define U_ON_SHMEM	1

  // the diagonal of a Hermitian matrix is real
  if (scalar_traits<T>::is_complex) {
    _if_ (myid<=n) { zero_imag(a(myid,myid)); }
    sync_on_cg<T,tile_size>();
  }

  TR el = static_cast<TR>(0);

  #pragma unroll 1
  for (int i=n; i>=2; i--) {
    const int l = i - 1;


#if U_ON_SHMEM
    T * const u_ = shmem;
    T u_myid = (myid<=l) ? a(min(myid,l),i) : ZERO;
#else
    T * const u_ = &a(1,i);
    T u_myid = (myid<=l) ? u(min(myid,l)) : ZERO;
#endif


    T beta; {
      TR anorm = norm(u_myid);
      sum_on_cg<TR,tile_size>(anorm);
      const T ul = u_myid - static_cast<T>(anorm = - Sign(Sqrt(anorm), real(u_myid)));
      beta = Reciprocal(flip0to1(conj(ul) * static_cast<T>(anorm)));
      const bool eee = (myid == l);
      __UPDATE__(el, anorm, eee);
      __UPDATE__(u_myid, ul, eee);
#if U_ON_SHMEM
      u(myid) = u_myid;
#else
      _if_ (myid<=l) { u(myid) = u_myid; }
#endif
    } bcast_on_cg<T,tile_size>(beta,l);


    T v_myid = ZERO; {
      const int lx1 = (l & 0x1);
      /*
       if(l%2==1){
         v(1) = a(1,1) * u(1)
       }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<k+0;j++){
           ajk0 = a(j,k+0)
           v(j) += ajk0 * u(k+0)
           v(k+0) += ajk0 * u(j)
	 }
         v(k+0) += a(k+0,k+0) * u(k+0)
         for(j=1;j<k+1;j++){
           ajk1 = a(j,k+1)
           v(j) += ajk1 * u(k+1)
           v(k+1) += ajk1 * u(j)
         }
         v(k+1) += a(+1k,k+1) * u(k+1)
       }
       */
      /*
       if(l%2==1){ v(1) = a(1,1) * u(1) }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<k+1;j++){
           ajk0 = a(j,k+0)
           ajk1 = a(j,k+1)
           if(j<k) v(j) += ajk0 * u(k+0)
           v(j) += ajk1 * u(k+1)
           v(k+0) += ajk0 * u(j)
           v(k+1) += ajk1 * u(j)
         }
         v(k+1) += a(k+1,k+1) * u(k+1)
       }
       */
      /*
       if(l%2==1){ v(1) = a(1,1) * u(1) }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<=k+1;j++){
           ajk0 = (j<=k) ? a(j,k+0) : 0
           ajk1 = (j<=k+1) ? a(j,k+1) : 0
           if(j<=k+0) v(j) += ajk0 * u(k+0)
           if(j<=k+0) v(j) += ajk1 * u(k+1)
           v(k+0) += ajk0 * ((j<=k-1) ? u(j) : 0)
           v(k+1) += ajk1 * u(j)
         }
       }
       */
      /*
       if(l%2==1){ v(1) = a(1,1) * u(1) }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<=k+1;j++){
           ajk0 = (j<=k) ? a(j,k+0) : 0
           ajk1 = (j<=k+1) ? a(j,k+1) : 0
           vj = v(j) + ajk0 * u(k+0) + ajk1 * u(k+1)
           if(j<=k+0) v(j) = vj
           vk0(j) = ((j<=k-1) ? ajk0 * u(j) : 0)
           vk1(j) = ajk1 * u(j)
         }
	 red2_on {vk0(:),vk1(:)} => v0 = {0,...,sum(vk0),sum(vk1),0,...,0}
	 v(:) += v0(:)
       }
       */
      _if_ (lx1 != 0) {
        const T vj = static_cast<T>(real(*a_)) * u_myid;
        _if_ (myid==lx1) { v_myid = vj; }
      }
      // min(myid,nm) is added due to a guard for out-of-bound
      T *ajk_ptr = &a(min(myid,nm),1+lx1);
      const int k0 = 1+lx1;
      const int kx = l-myid;
      #pragma unroll 1
      for (int k=k0, km=k-myid; km<=kx; k+=2, km+=2, ajk_ptr+=(2*nm)) {
        const T ajk0 = *(ajk_ptr+0*nm);
        const bool eee = (km>=-1); // k >= myid
        const T ajk1 = __MASK__(*(ajk_ptr+1*nm), eee);

        const T vj = (v_myid + ajk0 * u(k+0)) + ajk1 * u(k+1);
        __UPDATE__(v_myid, vj, km>=0); // k >= myid
        const bool fff = (km>=+1); // k >= myid+1
        // v(k) += conj(a(j,k)) * u(j)   (a(k,j) = conj(a(j,k)))
        const T vk0 = __MASK__(conj(ajk0) * u_myid, fff);
        const T vk1 = conj(ajk1) * u_myid;

	// -km = myid - k -> { myid-(1+lx1), myid-(3+lx1), ... }
	//  myid = {1+lx1, 2+lx1}, {3+lx1,4+lx1}, ...
	//       = {k0+0,k0+1}, {k0+2,k0+3}, ...

        const T vkk = red2_on_cg<T,tile_size>(vk0, vk1, -km);
        v_myid += vkk;
      }
#if U_ON_SHMEM
      _if_ (myid<=l) { *ajk_ptr = u_myid; }
#endif
    }


#if U_ON_SHMEM
    T * const v_ = shmem + tile_size;
#else
    T * const v_ = shmem;
#endif
    {
      v_myid *= beta;
      T alpha = conj(v_myid) * u_myid;
      sum_on_cg<T,tile_size>(alpha);
      alpha *= (beta * static_cast<T>(0.5));
      v_myid += alpha * u_myid;
      v(myid) = v_myid;
    } sync_on_cg<T,tile_size>();


    {
      T * ajk_ptr = &a(min(myid,nm),1);
      const int lx1 = (l & 0x1);
      _if_ (lx1) {
        T ajk = *ajk_ptr;
        ajk += u_myid * conj(v_myid);
        ajk += v_myid * conj(u_myid);
        zero_imag(ajk);
        _if_ (myid==lx1) { *ajk_ptr = ajk; }
        ajk_ptr += nm;
      }
      #pragma unroll 1
      for (int k=1+lx1; k<=l; k+=2) {
        T ajk0 = *(ajk_ptr);
        // a(j,k) += u(j)*conj(v(k)) + v(j)*conj(u(k)),  j = myid
        ajk0 += u_myid * conj(v(k+0));
        T ajk1 = *(ajk_ptr+nm);
        ajk1 += u_myid * conj(v(k+1));
        ajk0 += v_myid * conj(u(k+0));
        _if_ (myid==k+0) { zero_imag(ajk0); }
        _if_ (myid<=k+0) { *ajk_ptr = ajk0; } ajk_ptr += nm;
        ajk1 += v_myid * conj(u(k+1));
        _if_ (myid==k+1) { zero_imag(ajk1); }
        _if_ (myid<=k+1) { *ajk_ptr = ajk1; } ajk_ptr += nm;
      }
    }

  } sync_on_cg<T,tile_size>();

  _if_ (myid<=n) {
    const int j = (myid<n) ? myid+1 : 1;
    T * ajj_ptr = a_+(j-1)*(nm+1);
    d(j) = real(*ajj_ptr);
    e(myid) = el;
    *ajj_ptr = static_cast<T>(el);
  }

  // d and e are on shmem, they are passed to tql as well.

#undef        a
#undef        u
#undef        v
#undef        d
#undef        e
  sync_on_cg<T,tile_size>();
}

