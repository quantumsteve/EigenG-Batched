#define	TIMER		0
#if defined(__NVCC__)
#define	USE_IMTQL	0
#define DO_PREFETCH	0
#define	DO_SORT		1
#endif
#if defined(__HIPCC__)
#define	USE_IMTQL	0
#define DO_PREFETCH	0
#define	DO_SORT		1
#endif
#include "gpu_arch.h"


//--
#include "misc_gpu.hpp"
#include "eigen_GPU_batch.hpp"
//--
#include "hhsy2tr.hpp"
#include "hhsy2tr_tiled.hpp"
#include "tred1_tiled.hpp"
//--
#include "imtql2.hpp"
#include "imtql2_tiled.hpp"
#include "tql2.hpp"
#include "tql2_tiled.hpp"
//--
#include "hhtr2sy.hpp"
#include "hhtr2sy_tiled.hpp"
//--
#include "jac_tiled.hpp"
//--



//
// Launch bounds.  The default bound is the device maximum (1024 threads),
// which on AMD caps a kernel at 128 registers per thread (VGPR+AGPR, occupancy
// 4).  The batch kernels launch only WARP_GPU_SIZE*8 threads, so a tighter
// bound lets the compiler use up to 256 registers, at occupancy 2.
//
// Measured on MI100: complex double spills heavily at the default (scratch
// 350-1100 bytes/lane) and runs 1.3-1.6x faster with the tighter bound, while
// real float/double and complex float lose up to 40% at some sizes (their
// kernels did not spill, so occupancy 2 only costs).  Hence the bound is set
// per scalar type and, by default, only for complex double on HIP.
//   -DEIGENG_LAUNCH_BOUNDS=<threads>  force the bound on ALL types (experiments;
//                                     must be >= WARP_GPU_SIZE*8)
//   -DEIGENG_NO_CPLX_LB               HIP: no bound for complex double either
//   -DEIGENG_NO_LB                    no __launch_bounds__ attribute on any kernel
//                                     (the code generation before these switches)
//
#if defined(EIGENG_NO_LB)
// no attribute at all
#elif defined(EIGENG_LAUNCH_BOUNDS)
#define EIGENG_LB_ON 1
template <class T> struct eigen_lb { static constexpr int value = EIGENG_LAUNCH_BOUNDS; };
#elif (defined(__HIPCC__) || defined(EIGENG_PER_TYPE_LB)) && !defined(EIGENG_NO_CPLX_LB)
#define EIGENG_LB_ON 1
template <class T> struct eigen_lb { static constexpr int value = 1024; };   // = default
template <> struct eigen_lb< gpu_complex<double> > { static constexpr int value = WARP_GPU_SIZE*8; };
#endif
#if defined(EIGENG_LB_ON)
#define EIGENG_LB __launch_bounds__(eigen_lb<T>::value)
#else
#define EIGENG_LB
#endif

template <class T>
__global__ void
eigen_GPU_batch_n1_(const int L, const int nm, const int n, T *a_, real_t<T> *w_)
{
  const int pos = threadIdx.x+blockIdx.x*blockDim.x;
  if (pos>=L) return;

  const size_t len = nm*n;
  T * a = a_ + pos*len;
  real_t<T> * w = w_ + pos;

  const int step = blockDim.x*gridDim.x;
  #pragma unroll
  for (int id=pos; id<L; id+=step) {
    *w = real(*a);
    *a = static_cast<T>(1.0);
    a += len*step; w += n*step;
  }
}

template <class T, int tile_size>
__global__ void EIGENG_LB
eigen_GPU_batch_tiled_(const int L, const int nm, const int n, const int m, T *a_, real_t<T> *w_, T *wk_, const int PRELOAD_SLOT)
{
  typedef real_t<T> TR;
  const size_t blks = (blockDim.x*gridDim.x)/tile_size;
  const size_t len  = (long)nm*n;

#if 0
  const int    L_   = L;
  const size_t pos_ = (threadIdx.x+blockIdx.x*blockDim.x)/tile_size;
  const size_t pos  = pos_;
  const int    step = blks;
#else
  const int    L_   = (L-1)/blks+1;
  const size_t pos_ = (threadIdx.x+blockIdx.x*blockDim.x)/tile_size;
  const size_t pos  = pos_ * L_;
  const int    step = 1;
#endif

  // per-matrix work size in units of T (== len for real T)
  const size_t wlen = eigen_GPU_batch_wk_elems<T>(nm, n);

  T * a, * wk;
  TR * w;

#if DO_PREFETCH
  a = a_ + pos*len;
  #pragma unroll 1
  for (int id_=0; id_<PRELOAD_SLOT; id_+=step) {
    int id = id_ + pos;
    if ( id < L )
    prefetch_mat_cg<T, tile_size> (nm*n, a);
    a += len*step;
  }
#endif

  a  = a_  + pos*len;
  w  = w_  + pos*m;
  wk = wk_ + pos_*wlen;
  // real eigenvectors of the tridiagonal problem (Hermitian case: behind the
  // complex Z; real case: Z itself)
  TR * const yk = (TR *)(scalar_traits<T>::is_complex ? (wk + len) : wk);

  #pragma unroll 1
  for (int id_=0; id_<L_; id_+=step) {
    const int id = id_ + pos;
    const bool run = (id < L);

    SYNC_IF_NEEDED();
#if DO_PREFETCH
    if ( PRELOAD_SLOT > 0 && id+PRELOAD_SLOT*step < L )
    prefetch_mat_cg<T, tile_size> (nm*n, a+PRELOAD_SLOT*len*step);
#endif

    SYNC_IF_NEEDED();
#if 1
#if 1
    if(run) tred1_tiled_<T, tile_size> (nm, n, a);
#else
    const int mb = tile_size<=16?2:3; // 2*(mb+1) <= n
    const int lenu = nm*mb+(4+(nm*mb)%2);
    if (2*lenu<=n*nm) {
    T * u_=wk+0*lenu; // mb*nm+max(BLK_J,BLK_K)
    T * v_=wk+1*lenu; // mb*nm+max(BLK_J,BLK_K)
    if(run) hhsy2tr_tiled_<T, tile_size> (nm, n, a, mb, u_, v_);
    } else {
    if(run) tred1_tiled_<T, tile_size> (nm, n, a);
    }
#endif

    int return_flag = false;
    SYNC_IF_NEEDED();
#if USE_IMTQL
    if(run) return_flag = imtql2_tiled_<T, tile_size> (nm, n, w, wk);
#else
    if(run) return_flag = tql2_tiled_<TR, tile_size, T> (nm, n, w, yk);
#endif

    SYNC_IF_NEEDED();
    if (scalar_traits<T>::is_complex) {
      // Z := complex(y)
      if(run) if ( !return_flag ) {
        const int myid = threadIdx.x % tile_size + 1;
        if (myid <= n) {
          for (int j=0; j<n; j++) {
            wk[(myid-1)+(size_t)j*nm] = static_cast<T>(yk[(myid-1)+(size_t)j*nm]);
          }
        }
      }
      SYNC_IF_NEEDED();
    }

    if(run) if ( !return_flag ) hhtr2sy_tiled_<T, tile_size> (nm, n, a, wk);
#else
    if(run) jac_tiled<T, tile_size> ( a, wk, nm, n, w );
#endif

    a += len*step; w += n*step;
  }
}


template <class T>
__global__ void EIGENG_LB
eigen_GPU_batch_(const int L, const int nm, const int n, const int m, T *a_, real_t<T> *w_, real_t<T> *e_, T * wk_, const int PRELOAD_SLOT)
{
  typedef real_t<T> TR;
  const size_t blks = (blockDim.x*gridDim.x)/WARP_GPU_SIZE;
  const size_t len  = (long)nm*n;

#if 1
  const int    L_   = L;
  const size_t pos_ = (threadIdx.x+blockIdx.x*blockDim.x)/WARP_GPU_SIZE;
  const size_t pos  = pos_;
  const int    step = blks;
#else
  const int    L_   = (L-1)/blks+1;
  const size_t pos_ = (threadIdx.x+blockIdx.x*blockDim.x)/WARP_GPU_SIZE;
  const size_t pos  = pos_ * L_;
  const int    step = 1;
#endif

#if TIMER
  const int myid = (threadIdx.x % WARP_GPU_SIZE);
#endif

  // per-matrix work size in units of T (== len for real T)
  const size_t wlen = eigen_GPU_batch_wk_elems<T>(nm, n);

  T * a, * wk;
  TR * w, * e;

#if DO_PREFETCH
  a = a_ + pos*len;
  #pragma unroll 1
  for (int id_=0; id_<PRELOAD_SLOT; id_+=step) {
    int id = id_ + pos;
    if ( id < L )
    prefetch_mat_cg<T, WARP_GPU_SIZE> (nm*n, a);
    a += len*step;
  }
#endif

#if TIMER
  unsigned long t0, t1, t2, t3;
  unsigned long s0 = 0, s1 = 0, s2 = 0;
  unsigned long done = 0;
#endif

  a  = a_  + pos*len;
  w  = w_  + pos*n;
  wk = wk_ + pos_*wlen;
  e  = e_  + pos_*n;
  // real eigenvectors of the tridiagonal problem (Hermitian case: behind the
  // complex Z; real case: Z itself)
  TR * const yk = (TR *)(scalar_traits<T>::is_complex ? (wk + len) : wk);

  const int myid_w = (threadIdx.x % WARP_GPU_SIZE) + 1;

  #pragma unroll 1
  for (int id_=0; id_<L_; id_+=step) {
    const int id = id_ + pos;
    const bool run = (id < L);

#if DO_PREFETCH
    if ( PRELOAD_SLOT > 0 && id+PRELOAD_SLOT*step < L )
    prefetch_mat_cg<T, WARP_GPU_SIZE> (nm*n, a+PRELOAD_SLOT*len*step);
#endif

#if TIMER
    t0 = __global_timer__();
#endif
    // n >= 33 --> mb <= min(16,34/2-1) = 16
//    const int mb=2; // 2*(mb+1) <= n+1 && 2*mb <= WARP_GPU_SIZE
//    const int mb=3; // 2*(mb+1) <= n+1 && 2*mb <= WARP_GPU_SIZE
    const int mb=4; // 2*(mb+1) <= n+1 && 2*mb <= WARP_GPU_SIZE
//    const int mb=16; // 2*(mb+1) <= n+1 && 2*mb <= WARP_GPU_SIZE
    T * u_=wk+0*nm*(mb+1); // < mb*nm+max(BLK_J,BLK_K)
    T * v_=wk+1*nm*(mb+1); // < mb*nm+max(BLK_J,BLK_K)
    if(run) hhsy2tr_ (nm, n, a, w, e, mb, u_, v_);
#if TIMER
    t1 = __global_timer__();
    s0 = s0 + (t1 - t0);
#endif
    int return_flag = false;
#if USE_IMTQL
    if(run) return_flag = imtql2_ (nm, n, w, e, wk);
#else
    if(run) return_flag = tql2_<TR, T> (nm, n, w, e, yk);
#endif
    if ( return_flag ) break;
    if (scalar_traits<T>::is_complex) {
      // Z := complex(y)
      if(run) {
        for (int j=0; j<n; j++) {
          for (int i=myid_w; i<=n; i+=WARP_GPU_SIZE) {
            wk[(i-1)+(size_t)j*nm] = static_cast<T>(yk[(i-1)+(size_t)j*nm]);
          }
        }
      }
      sync_on_warp();
    }
#if TIMER
    t2 = __global_timer__();
    s1 = s1 + (t2 - t1);
#endif
    if(run) hhtr2sy_ (nm, n, a, m, wk, DO_SORT ? (int*)e : nullptr);
#if TIMER
    t3 = __global_timer__();
    s2 = s2 + (t3 - t2);
    done++;
#endif

    a += len*step; w += n*step;
  }

#if TIMER
  if (pos==0 && myid == 0) {
    double t;
    t = (double)s0*1e-9;
    printf("TRED1  : %le [s]: %le [GFLOPS]: %le [GB/s]\n",
		       t, 1e-9*done*4.*n*n*n/3/t, 1e-9*(double)done*sizeof(T)*n*n/2/t);
    t = (double)s1*1e-9;
    printf("TQL2   : %le [s]: %le [GFLOPS]: %le [GB/s]\n",
		       t, 1e-9*done*0., 1e-9*(double)done*sizeof(T)*n*(n+2)/t);
    t = (double)s2*1e-9;
    printf("TRBAK1 : %le [s]: %le [GFLOPS]: %le [GB/s]\n",
		       t, 1e-9*done*2.*n*n*n/t, 1e-9*(double)done*sizeof(T)*3*n*n/2/t);
  }
#endif
}


// n > WARP_GPU_SIZE.
// Work buffer: e (real, n per concurrent matrix), then wk (wlen elements of T
// per concurrent matrix).  For real T this is the original layout.
template <class T>
__host__ gpuError_t
eigen_GPU_batch_large_RUN(const int L, const int nm, const int n, const int m, T * a, real_t<T> * w, T * wk_,
                          const int numTB, const int numTH, const int numGR, const int sizeSH,
                          const int PRELOAD_SLOT, const gpuStream_t stream)
{
    typedef real_t<T> TR;
    const size_t cnt = min(numTB*numGR,L);
    TR *e = (TR *)wk_;
    T *wk = wk_ + (n*cnt*sizeof(TR) + sizeof(T) - 1)/sizeof(T);
#if defined(__NVCC__)
    cudaFuncSetAttribute( eigen_GPU_batch_ <T>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
    eigen_GPU_batch_ <T> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, e, wk, PRELOAD_SLOT);
    return gpuSuccess;
}
template <class T>
__host__ gpuError_t
eigen_GPU_batch_RUN(const int L, const int nm, const int n, const int m, T * a, real_t<T> * w, T * wk_, const gpuStream_t stream)
{
  if (L<1||nm<n||n<1||m<1||m>n) return gpuErrorInvalidValue;

  gpuError_t err_code;
  int grpSZ=0, numTB=0, numGR=0, numTH=0, sizeSH=0, sizeL2=0;
  err_code = eigen_GPU_batch_get_Launch_params<T>(L, n, grpSZ, numTB, numGR, numTH, sizeSH, sizeL2);
  if (err_code != gpuSuccess) return err_code;
#if defined(EIGENG_LB_ON)
  if (numTH > eigen_lb<T>::value) return gpuErrorInvalidValue;
#endif

#if DO_PREFETCH
  const int len = nm*n;
  const int step = (numTH*numTB)/grpSZ;
  const int ELE = sizeof(T)*len*step;
//  const int lwk = sizeof(T)*(n+(n*nm))*min(numTB*numGR,L);
//  const int PRELOAD_SLOT = max(sizeL2-lwk,0)/ELE;
  const int PRELOAD_SLOT = min(L,(sizeL2-1)/ELE);
#else
  const int PRELOAD_SLOT = 0;
#endif

  if (n == 1) {
#if defined(__NVCC__)
    cudaFuncSetAttribute( eigen_GPU_batch_n1_ <T>,
		    cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
    eigen_GPU_batch_n1_ <T> <<<numTB, numTH, 0, stream>>> (L, nm, n, a, w);
  } if ( n <= WARP_GPU_SIZE ) {
    switch ( grpSZ ) {
    case 4:
#if defined(__NVCC__)
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,4>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 4> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
    case 8:
#if defined(__NVCC__)
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,8>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 8> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
    case 16:
#if defined(__NVCC__)
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,16>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 16> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
    case 32:
#if defined(__NVCC__)
    default:
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,32>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 32> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
#if WARP_GPU_SIZE==64
    case 64:
    default:
      eigen_GPU_batch_tiled_ <T, 64> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
#endif
    }
  } else {
    return eigen_GPU_batch_large_RUN<T>(L, nm, n, m, a, w, wk_, numTB, numTH, numGR, sizeSH,
        PRELOAD_SLOT, stream);
  }

  return gpuSuccess;
}

extern "C" {

__host__ void
eigen_GPU_batch_DP(const int L, const int nm, const int n, const int m, double * a, double * w, double *wk, const gpuStream_t stream)
{
  if ( wk == NULL ) {
    if ( n > 1 ) {
      size_t lwork;
      eigen_GPU_batch_BufferSize <double>(L, nm, n, m, NULL, NULL, &lwork);
      *((size_t *)w) = lwork;
    } else {
      *((int64_t *)w) = 0;
    }
  } else {
    eigen_GPU_batch_RUN <double>(L, nm, n, m, a, w, wk, stream);
  }
}

__host__ void
eigen_GPU_batch_FP(const int L, const int nm, const int n, const int m, float * a, float * w, float *wk, const gpuStream_t stream)
{
  if ( wk == NULL ) {
    if ( n > 1 ) {
      size_t lwork;
      eigen_GPU_batch_BufferSize <float>(L, nm, n, m, NULL, NULL, &lwork);
      *((size_t *)w) = lwork;
    } else {
      *((int32_t *)w) = 0;
    }
  } else {
    eigen_GPU_batch_RUN <float>(L, nm, n, m, a, w, wk, stream);
  }
}

__host__ void
eigen_GPU_batch_ZP(const int L, const int nm, const int n, const int m, gpuDoubleComplex * a, double * w, gpuDoubleComplex *wk, const gpuStream_t stream)
{
  typedef gpu_complex<double> T;
  if ( wk == NULL ) {
    if ( n > 1 ) {
      size_t lwork;
      eigen_GPU_batch_BufferSize <T>(L, nm, n, m, NULL, NULL, &lwork);
      *((size_t *)w) = lwork;
    } else {
      *((int64_t *)w) = 0;
    }
  } else {
    eigen_GPU_batch_RUN <T>(L, nm, n, m, (T *)a, w, (T *)wk, stream);
  }
}

__host__ void
eigen_GPU_batch_CP(const int L, const int nm, const int n, const int m, gpuFloatComplex * a, float * w, gpuFloatComplex *wk, const gpuStream_t stream)
{
  typedef gpu_complex<float> T;
  if ( wk == NULL ) {
    if ( n > 1 ) {
      size_t lwork;
      eigen_GPU_batch_BufferSize <T>(L, nm, n, m, NULL, NULL, &lwork);
      *((size_t *)w) = lwork;
    } else {
      *((int32_t *)w) = 0;
    }
  } else {
    eigen_GPU_batch_RUN <T>(L, nm, n, m, (T *)a, w, (T *)wk, stream);
  }
}

//__host__ void
//eigen_GPU_batch_HP(const int L, const int nm, const int n, const int m, half * a, half * w, half *wk)
//{
//  eigen_GPU_batch_RUN <half>(L, nm, n, m, a, w, wk);
//}

}
