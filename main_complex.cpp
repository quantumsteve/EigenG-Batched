//
// Accuracy test of the complex Hermitian batched eigensolver
// (eigen_GPU_batch_ZP / eigen_GPU_batch_CP).
//
// For every size and matrix kind it checks, on the host in double precision,
//   resid = max_j || A v_j - w_j v_j ||_2 / (n eps ||A||_F)
//   orth  = max_jk |(V^H V - I)_jk|        / (n eps)
// plus ascending order of w and absence of NaN/Inf.  n orthonormal vectors
// with small residuals form the complete eigen-decomposition, so no
// reference solver is needed.
//
// usage: ./a.out-complex [poison]
//   poison: fill the strictly lower triangle of the input with NaN to verify
//           that only the upper triangle is referenced.
//
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <random>
#include <vector>
#include <limits>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "gpu_arch.h"
#include "eigen_GPU_batch.hpp"

template <class R> struct DevComplex;
template <> struct DevComplex<float>  { typedef gpuFloatComplex  type; };
template <> struct DevComplex<double> { typedef gpuDoubleComplex type; };

enum Kind { RANDOM = 0, REAL_SYM, DIAGONAL, RANK1_PLUS_I, NUM_KINDS };
static const char * kind_name[] = { "random", "real-sym", "diagonal", "rank1+I" };

typedef std::complex<double> cd;

// A is n x n column-major (ld nm), full Hermitian
static void
make_matrix(std::vector<cd> &A, const int nm, const int n, const Kind kind, std::mt19937 &gen)
{
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  A.assign((size_t)nm*n, cd(0,0));
  if (kind == RANDOM || kind == REAL_SYM) {
    for (int k=0; k<n; k++) {
      A[k+(size_t)nm*k] = cd(U(gen), 0);
      for (int j=0; j<k; j++) {
        const cd x(U(gen), kind==RANDOM ? U(gen) : 0.0);
        A[j+(size_t)nm*k] = x;
        A[k+(size_t)nm*j] = std::conj(x);
      }
    }
  } else if (kind == DIAGONAL) {
    for (int k=0; k<n; k++) A[k+(size_t)nm*k] = cd(U(gen), 0);
  } else {
    std::vector<cd> u(n);
    for (int k=0; k<n; k++) u[k] = cd(U(gen), U(gen));
    for (int k=0; k<n; k++) {
    for (int j=0; j<n; j++) {
      A[j+(size_t)nm*k] = u[j]*std::conj(u[k]) + (j==k ? cd(1,0) : cd(0,0));
    }}
  }
}

template <class R>
static bool
run_case(const int L, const int n, const Kind kind, const bool poison,
         double &resid_max, double &orth_max)
{
  typedef typename DevComplex<R>::type DC;
  const int nm = n + (n&0x1);
  const int m  = n;
  const R eps = std::numeric_limits<R>::epsilon();
  const R NaN = std::numeric_limits<R>::quiet_NaN();

  std::mt19937 gen(12345u + 7919u*n + 31u*kind);
  std::vector< std::vector<cd> > A(L);
  std::vector<R> a_h(2*(size_t)L*nm*n, 0);
  for (int id=0; id<L; id++) {
    make_matrix(A[id], nm, n, kind, gen);
    for (int k=0; k<n; k++) {
    for (int j=0; j<nm; j++) {
      cd x = (j<n) ? A[id][j+(size_t)nm*k] : cd(0,0);
      // the reference must be the matrix the GPU actually sees (rounded to R)
      if (j<n) { x = cd((double)(R)x.real(), (double)(R)x.imag()); A[id][j+(size_t)nm*k] = x; }
      if (poison && j>k && j<n) x = cd(NaN, NaN);
      const size_t p = 2*((size_t)id*nm*n + j + (size_t)nm*k);
      a_h[p] = (R)x.real(); a_h[p+1] = (R)x.imag();
    }}
  }

  DC *a_d = NULL, *wk_d = NULL; R *w_d = NULL;
  const size_t lena = sizeof(DC)*(size_t)L*nm*n, lenw = sizeof(R)*(size_t)L*m;
  size_t lenwk = 0;
  if (gpuMalloc((void**)&a_d, lena) != gpuSuccess ||
      gpuMalloc((void**)&w_d, lenw) != gpuSuccess) { printf("gpuMalloc failed\n"); return false; }
  if (eigen_GPU_batch_BufferSize<DC>(L, nm, n, m, a_d, w_d, &lenwk) != gpuSuccess ||
      gpuMalloc((void**)&wk_d, lenwk) != gpuSuccess) { printf("work buffer failed\n"); return false; }
  gpuMemcpy(a_d, a_h.data(), lena, gpuMemcpyHostToDevice);

  gpuStream_t stream;
  gpuStreamCreate(&stream);
  eigen_GPU_batch<DC>(L, nm, n, m, a_d, w_d, wk_d, stream);
  gpuDeviceSynchronize();

  std::vector<R> v_h(2*(size_t)L*nm*n), w_h((size_t)L*m);
  gpuMemcpy(v_h.data(), a_d, lena, gpuMemcpyDeviceToHost);
  gpuMemcpy(w_h.data(), w_d, lenw, gpuMemcpyDeviceToHost);
  gpuStreamDestroy(stream);
  gpuFree(a_d); gpuFree(w_d); gpuFree(wk_d);

  bool ok = true;
  resid_max = orth_max = 0;
  for (int id=0; id<L; id++) {
    const std::vector<cd> &Ai = A[id];
    const R *w = &w_h[(size_t)id*m];
    const R *v = &v_h[2*(size_t)id*nm*n];
    auto V = [&](int j, int k) { const size_t p = 2*((size_t)j + (size_t)nm*k); return cd(v[p], v[p+1]); };

    double fro = 0;
    for (int k=0; k<n; k++) for (int j=0; j<n; j++) fro += std::norm(Ai[j+(size_t)nm*k]);
    fro = sqrt(fro); if (fro == 0) fro = 1;

    for (int j=0; j<n; j++) {
      if (!std::isfinite((double)w[j])) ok = false;
      if (j>0 && w[j] < w[j-1]) ok = false;               // ascending
    }
    for (int k=0; k<n; k++) {
      double r2 = 0;
      for (int j=0; j<n; j++) {
        cd s(0,0);
        for (int l=0; l<n; l++) s += Ai[j+(size_t)nm*l] * V(l,k);
        s -= (double)w[k] * V(j,k);
        r2 += std::norm(s);
      }
      const double r = sqrt(r2)/(n*(double)eps*fro);
      if (!(r == r)) ok = false; else if (r > resid_max) resid_max = r;
      for (int l=0; l<=k; l++) {
        cd s(0,0);
        for (int j=0; j<n; j++) s += std::conj(V(j,l)) * V(j,k);
        s -= (l==k ? cd(1,0) : cd(0,0));
        const double o = std::abs(s)/(n*(double)eps);
        if (!(o == o)) ok = false; else if (o > orth_max) orth_max = o;
      }
    }
  }
  if (resid_max > 1e3 || orth_max > 1e3) ok = false;
  return ok;
}

template <class R>
static int
run_all(const char *name, const bool poison)
{
  int nfail = 0;
  const int L = 257;   // not a multiple of the group/block sizes
  const int nums[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 0 };
  printf(">> %s complex Hermitian, batch=%d%s\n", name, L, poison ? ", NaN in lower triangle" : "");
  for (int i=0; nums[i] > 0; i++) {
    const int n = nums[i];
    if (n > WARP_GPU_SIZE) break;   // larger sizes are not supported for complex yet
    for (int k=0; k<NUM_KINDS; k++) {
      double resid, orth;
      const bool ok = run_case<R>(L, n, (Kind)k, poison, resid, orth);
      printf("  n=%3d %-9s resid=%8.2e orth=%8.2e  %s\n", n, kind_name[k], resid, orth, ok ? "ok" : "FAIL");
      if (!ok) nfail++;
    }
  }
  return nfail;
}

int
main(int argc, char **argv)
{
  const bool poison = (argc > 1 && strcmp(argv[1], "poison") == 0);
  gpuSetDevice(0);
  int nfail = 0;
  nfail += run_all<float>("float", poison);
  nfail += run_all<double>("double", poison);
  printf("%s (%d failing cases)\n", nfail ? "FAILED" : "PASSED", nfail);
  return nfail ? 1 : 0;
}
