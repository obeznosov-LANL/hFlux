//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#pragma once

#include "common.hpp"
#include "StructuredLocator.hpp"

template<int m, int swidth,  typename ExecSpace, int ndims = 3>
struct FieldData {

  StructuredLocator fd_locator;
  StructuredLocator hermite_locator;

  Kokkos::View<Real***, Kokkos::LayoutRight, ExecSpace> data;
  Kokkos::View<Real*****, Kokkos::LayoutLeft, ExecSpace> hermite_data;
  Kokkos::View<Real****, Kokkos::LayoutLeft, ExecSpace> psi_data;

  FieldData(int nR_data, int nZ_data,
                      Real R0, Real Z0, Real dR, Real dZ)
      : fd_locator(R0, Z0, dR, dZ, nR_data - 1, nZ_data - 1),
        hermite_locator(makeHermiteLocator<swidth>(fd_locator)),
        data("data", nR_data, nZ_data, ndims),
        hermite_data("hermite_data",
                      2 * m + 3, 2 * m + 3, ndims,
                      hermite_locator.nR,
                      hermite_locator.nZ),
        psi_data("psi_data",
                  2 * m + 3, 2 * m + 3,
                  hermite_locator.nR,
                  hermite_locator.nZ) {}
};
