#ifndef __HEADER_COMPLEX_GPU_HPP__
#define __HEADER_COMPLEX_GPU_HPP__

//
// gpu_complex<R>: a minimal complex scalar for device code.
//
// It is layout-compatible with cuComplex/cuDoubleComplex/hipFloatComplex/
// hipDoubleComplex (two contiguous R, 2*sizeof(R)-aligned), so API pointers
// of the vendor types can be reinterpret_cast to gpu_complex<R>*.
// A dedicated type is used (instead of the vendor ones) so that operator
// semantics are identical on CUDA and HIP; HIP's float2 has component-wise
// operator*, which would silently be wrong for complex multiplication.
//
// Expects __host__/__device__ to be defined by the CUDA/HIP headers.
//
#include <math.h>
#include <type_traits>

template <class R>
struct alignas(2*sizeof(R)) gpu_complex {
  typedef R value_type;
  R re, im;

  __host__ __device__ __forceinline__
  gpu_complex() : re(0), im(0) {}
  // implicit: allows static_cast<gpu_complex<R>>(0) and mixed real literals
  __host__ __device__ __forceinline__
  gpu_complex(const R r) : re(r), im(0) {}
  __host__ __device__ __forceinline__
  gpu_complex(const R r, const R i) : re(r), im(i) {}

  __host__ __device__ __forceinline__
  gpu_complex & operator+=(const gpu_complex &y) { re += y.re; im += y.im; return *this; }
  __host__ __device__ __forceinline__
  gpu_complex & operator-=(const gpu_complex &y) { re -= y.re; im -= y.im; return *this; }
  __host__ __device__ __forceinline__
  gpu_complex & operator*=(const gpu_complex &y) {
    const R r = re*y.re - im*y.im;
    const R i = re*y.im + im*y.re;
    re = r; im = i; return *this;
  }
  __host__ __device__ __forceinline__
  gpu_complex & operator*=(const R y) { re *= y; im *= y; return *this; }
};

template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator+(const gpu_complex<R> &x, const gpu_complex<R> &y)
{ return gpu_complex<R>(x.re+y.re, x.im+y.im); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator-(const gpu_complex<R> &x, const gpu_complex<R> &y)
{ return gpu_complex<R>(x.re-y.re, x.im-y.im); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator-(const gpu_complex<R> &x)
{ return gpu_complex<R>(-x.re, -x.im); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator*(const gpu_complex<R> &x, const gpu_complex<R> &y)
{ return gpu_complex<R>(x.re*y.re - x.im*y.im, x.re*y.im + x.im*y.re); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator*(const R a, const gpu_complex<R> &y)
{ return gpu_complex<R>(a*y.re, a*y.im); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator*(const gpu_complex<R> &x, const R a)
{ return gpu_complex<R>(x.re*a, x.im*a); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator/(const gpu_complex<R> &x, const R a)
{ return gpu_complex<R>(x.re/a, x.im/a); }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> operator/(const gpu_complex<R> &x, const gpu_complex<R> &y)
{
  // x * conj(y) / |y|^2
  const R d = y.re*y.re + y.im*y.im;
  return gpu_complex<R>((x.re*y.re + x.im*y.im)/d, (x.im*y.re - x.re*y.im)/d);
}
template <class R> __host__ __device__ __forceinline__
bool operator==(const gpu_complex<R> &x, const gpu_complex<R> &y)
{ return (x.re == y.re) && (x.im == y.im); }
template <class R> __host__ __device__ __forceinline__
bool operator!=(const gpu_complex<R> &x, const gpu_complex<R> &y)
{ return !(x == y); }

//
// Real/imag/conj/norm.  The real-type overloads let generic code written for
// Hermitian matrices (conj(x)*y, real(x), ...) also instantiate for T=float/double.
//
template <class R> __host__ __device__ __forceinline__
R real(const gpu_complex<R> &x) { return x.re; }
template <class R> __host__ __device__ __forceinline__
R imag(const gpu_complex<R> &x) { return x.im; }
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> conj(const gpu_complex<R> &x) { return gpu_complex<R>(x.re, -x.im); }
// squared modulus (always real)
template <class R> __host__ __device__ __forceinline__
R norm(const gpu_complex<R> &x) { return x.re*x.re + x.im*x.im; }

__host__ __device__ __forceinline__ float  real(const float x)  { return x; }
__host__ __device__ __forceinline__ double real(const double x) { return x; }
__host__ __device__ __forceinline__ float  imag(const float)    { return 0.0f; }
__host__ __device__ __forceinline__ double imag(const double)   { return 0.0; }
__host__ __device__ __forceinline__ float  conj(const float x)  { return x; }
__host__ __device__ __forceinline__ double conj(const double x) { return x; }
__host__ __device__ __forceinline__ float  norm(const float x)  { return x*x; }
__host__ __device__ __forceinline__ double norm(const double x) { return x*x; }

// force exact reality (Hermitian diagonal); no-op for real scalars
template <class R> __host__ __device__ __forceinline__
void zero_imag(gpu_complex<R> &x) { x.im = R(0); }
__host__ __device__ __forceinline__ void zero_imag(float &)  {}
__host__ __device__ __forceinline__ void zero_imag(double &) {}

//
// |z| (hypot-based: no overflow/underflow in re^2+im^2)
//
__host__ __device__ __forceinline__ float  gpu_hypot_(const float a,  const float b)  { return hypotf(a, b); }
__host__ __device__ __forceinline__ double gpu_hypot_(const double a, const double b) { return hypot(a, b); }

template <class R> __host__ __device__ __forceinline__
R Abs(const gpu_complex<R> &x) { return gpu_hypot_(x.re, x.im); }

//
// Sign(x, y) for real x and complex y: |x| * y/|y|  (the phase of y with
// magnitude |x|).  Reduces to copysign(|x|, y) when y is real, and returns
// |x| when y == 0, consistent with the real Sign(x,y) for y >= 0 side.
//
template <class R> __host__ __device__ __forceinline__
gpu_complex<R> Sign(const R x, const gpu_complex<R> &y)
{
  const R ay = gpu_hypot_(y.re, y.im);
  const R ax = (x < R(0)) ? -x : x;
  return (ay == R(0)) ? gpu_complex<R>(ax, R(0))
                      : gpu_complex<R>(ax*(y.re/ay), ax*(y.im/ay));
}

#endif
