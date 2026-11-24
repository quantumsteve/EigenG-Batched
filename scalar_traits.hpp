#ifndef __HEADER_SCALAR_TRAITS_HPP__
#define __HEADER_SCALAR_TRAITS_HPP__

//
// Separates the matrix scalar type (TA) from the real type (TR) used for
// eigenvalues and the tridiagonal d/e vectors.
//   real float/double          : TA == TR
//   complex (Hermitian) matrix : TA = complex<TR>, TR = float/double
//
#if defined(__NVCC__)
#include <cuComplex.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_complex.h>
#endif

template <class T>
struct scalar_traits {
  typedef T real_type;
  static constexpr bool is_complex = false;
};

#if defined(__NVCC__)
template <> struct scalar_traits<cuComplex> {
  typedef float real_type;
  static constexpr bool is_complex = true;
};
template <> struct scalar_traits<cuDoubleComplex> {
  typedef double real_type;
  static constexpr bool is_complex = true;
};
#endif
#if defined(__HIPCC__)
template <> struct scalar_traits<hipFloatComplex> {
  typedef float real_type;
  static constexpr bool is_complex = true;
};
template <> struct scalar_traits<hipDoubleComplex> {
  typedef double real_type;
  static constexpr bool is_complex = true;
};
#endif

template <class T>
using real_t = typename scalar_traits<T>::real_type;

#endif
