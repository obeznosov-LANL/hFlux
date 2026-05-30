#pragma once
#include "common.hpp"

template<class T>
KOKKOS_INLINE_FUNCTION
void evalTaylor(Real& val, Real x, Real y, const T pcofs) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);

  val = 0.0;
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
        val += mon * pcofs(i, j);
        mon *= y;
    }
    sclx *= x;
  }
}

template<class T>
KOKKOS_INLINE_FUNCTION
void evalTaylor3(Dim3& vals, Real x, Real y, const T pcofs) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);
  KOKKOS_ASSERT(pcofs.extent_int(2) >= 3);

  vals = {};
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
      for (int d = 0; d < 3; ++d) {
         vals[d] += mon * pcofs(i, j, d);
       }
       mon *= y;
    }
    sclx *= x;
  }
}
