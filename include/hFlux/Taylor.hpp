#pragma once
#include "common.hpp"

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylor(Real& val, Real x, Real y, CoeffView pcofs) {

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

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorScalar(Real& val, Real x, Real y, CoeffView pcofs, int component = 0) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);

  val = 0.0;
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
        val += mon * pcofs(i, j, component);
        mon *= y;
    }
    sclx *= x;
  }
}

template<int ncomp, class Out, class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylorN(Out& vals, Real x, Real y, CoeffView pcofs, int component0 = 0) {

  const int Px = pcofs.extent_int(0);
  const int Py = pcofs.extent_int(1);
  KOKKOS_ASSERT(pcofs.extent_int(2) >= component0 + ncomp);

  vals = {};
  Real sclx = 1.0;
  for (int i = 0; i < Px; ++i) {
    Real mon = sclx;
    for (int j = 0; j < Py; ++j) {
      for (int d = 0; d < ncomp; ++d) {
         vals[d] += mon * pcofs(i, j, component0 + d);
       }
       mon *= y;
    }
    sclx *= x;
  }
}

template<class CoeffView>
KOKKOS_INLINE_FUNCTION
void evalTaylor3(Dim3& vals, Real x, Real y, CoeffView pcofs, int component0 = 0) {

  evalTaylorN<3, Dim3, CoeffView>(vals, x, y, pcofs, component0);
}
