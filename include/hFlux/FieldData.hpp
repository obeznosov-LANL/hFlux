#pragma once

#include "common.hpp"
#include "StructuredLocator.hpp"

template<int m, int swidth, typename ExecSpace>
struct FieldData {
  static constexpr int ndims = 3;

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
