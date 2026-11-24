#ifndef __HEADER_EIGEN_GPU_BATCH_H__
#define __HEADER_EIGEN_GPU_BATCH_H__

#include <stdio.h>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__NVCC__)
#include <cuComplex.h>
typedef cuComplex       gpuFloatComplex;
typedef cuDoubleComplex gpuDoubleComplex;
#endif
#if defined(__HIPCC__)
#include <hip/hip_complex.h>
typedef hipFloatComplex  gpuFloatComplex;
typedef hipDoubleComplex gpuDoubleComplex;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_arch.h"

//
// if wk == NULL && n > 1 then ((size_t)w)[0] returns the length of
// work buffer required in the batch operation with an accordance to
// the matrix size and GPU architecture.
//

void
eigen_GPU_batch_DP(const int L, const int nm, const int n, const int m, double * a, double * w, double *wk, const gpuStream_t stream);

void
eigen_GPU_batch_FP(const int L, const int nm, const int n, const int m, float * a, float * w, float *wk, const gpuStream_t stream);

//
// Complex Hermitian (upper triangle of a is referenced).  w holds real
// eigenvalues.  Only n <= WARP_GPU_SIZE is supported for now.
// wk == NULL query works as above, with ((size_t)w)[0] as the result.
//
void
eigen_GPU_batch_ZP(const int L, const int nm, const int n, const int m, gpuDoubleComplex * a, double * w, gpuDoubleComplex *wk, const gpuStream_t stream);

void
eigen_GPU_batch_CP(const int L, const int nm, const int n, const int m, gpuFloatComplex * a, float * w, gpuFloatComplex *wk, const gpuStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

