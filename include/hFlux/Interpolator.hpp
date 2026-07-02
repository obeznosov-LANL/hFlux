#pragma once

#include "common.hpp"
#include "FiniteDifferenceWeights.hpp"
#include "StructuredLocator.hpp"

template <class View, class Scalar>
KOKKOS_INLINE_FUNCTION
void eval_nonconst_at_half(const View& hermite_data,
                           const int idR,
                           const int component,
                           const int iRcell,
                           const int iZcell,
                           const int Pz,
                           Scalar& sum_plus,
                           Scalar& sum_minus)
{
  // sum_plus  = sum_{k>=1} a_k * (+0.5)^k
  // sum_minus = sum_{k>=1} a_k * (-0.5)^k
  sum_plus  = Scalar(0);
  sum_minus = Scalar(0);

  Scalar p_plus  = Scalar(0.5);   // (+0.5)^1
  Scalar p_minus = Scalar(-0.5);  // (-0.5)^1
  for (int k = 1; k < Pz; ++k) {
    const Scalar ak = hermite_data(idR, k, component, iRcell, iZcell);
    sum_plus  += ak * p_plus;
    sum_minus += ak * p_minus;
    p_plus  *= Scalar(0.5);
    p_minus *= Scalar(-0.5);
  }
};

template <class View, class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar eval_z_at(const View& data,
                 const int idR,
                 const int component,
                 const int iRcell,
                 const int iZcell,
                 const int Pz,
                 const Scalar zeta)
{
  Scalar sum = Scalar(0);
  Scalar p = Scalar(1);
  for (int idZ = 0; idZ < Pz; ++idZ) {
    sum += data(idR, idZ, component, iRcell, iZcell) * p;
    p *= zeta;
  }

  return sum;
};

template<int m, int swidth, class ViewData, class ViewHermiteData>
KOKKOS_INLINE_FUNCTION
void computeDerivativesStencil(ViewData view_data, ViewHermiteData view_hermite_data, const Real ratioR, const Real ratioZ) {
  KOKKOS_ASSERT(view_data.rank == 2);
  KOKKOS_ASSERT(view_data.extent(0) == swidth);
  KOKKOS_ASSERT(view_data.extent(1) == swidth);

  KOKKOS_ASSERT(view_hermite_data.rank == 2);
  KOKKOS_ASSERT(view_hermite_data.extent(0) == m+1);
  KOKKOS_ASSERT(view_hermite_data.extent(1) == m+1);

  constexpr auto D = fdw<swidth>();

  Real sclx = 1.0;
  for (int idx = 0; idx < m+1; ++idx)
  {
    Real scly = 1.0;
    for (int idy = 0; idy < m+1; ++idy)
    {
      auto& hdof = view_hermite_data(idx, idy);
      hdof = 0.0;
      for (int i = 0; i < swidth; ++i) {
        for (int j = 0; j < swidth; ++j) {
          hdof += D[idx][i] * D[idy][j] * view_data(i,j) * sclx * scly;
        }
      }
      scly *= ratioZ / (idy + 1);
    }
    sclx *= ratioR / (idx + 1);
  }
};

// Computes the finite difference derivates on point data collocated on some grid
// Stores the scaled derivatives (scaled) in hermite data, ready to be interpolated
template<int m, int swidth, class DataView, class HermiteView>
void compute_derivatives_grid (DataView data, HermiteView hermite_data,
                         const Real ratioR, const Real ratioZ) {

  static_assert(DataView::rank == 3);
  static_assert(HermiteView::rank == 5);

  KOKKOS_ASSERT(ratioR > 1.0);
  KOKKOS_ASSERT(ratioZ > 1.0);

  using exec_space = typename HermiteView::execution_space;

  const size_t n0 = hermite_data.extent(2);
  const size_t n1 = hermite_data.extent(3);
  const size_t n2 = hermite_data.extent(4);

  Kokkos::deep_copy(hermite_data, 0.0);

  Kokkos::parallel_for("compute_derivatives",
  Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0,0,0}, {n0, n1, n2}),
  KOKKOS_LAMBDA(int component, int i, int j) {
    for (int offx = 0; offx < 2; ++offx) {
      for (int offy = 0; offy < 2; ++offy) {
        int ii = (i + offx) * (swidth-1);
        int jj = (j + offy) * (swidth-1);
        int idx = (m+1) * offx;
        int idy = (m+1) * offy;

        auto data_stencil = Kokkos::subview(data,
            Kokkos::make_pair(ii, ii + swidth), Kokkos::make_pair(jj, jj + swidth), component);
        auto hermite_data_cell = Kokkos::subview(hermite_data,
            Kokkos::make_pair(idx, idx + m+1), Kokkos::make_pair(idy, idy + m+1), component, i, j);

        computeDerivativesStencil<m, swidth>(data_stencil, hermite_data_cell, ratioR, ratioZ);
      }
    }
  });
}

template<int m, class T>
KOKKOS_INLINE_FUNCTION
void interpolateInPlace1D(T& data) {
  KOKKOS_ASSERT(data.rank == 1);
  static const int sz = m+1;
  Kokkos::Array<Kokkos::Array<Real, 2*sz>, 2*sz> NT;

  for (int i = 0; i < sz; ++i)
    for (int idx = 0; idx < sz - i; ++idx)
    {
      NT[i][idx] = data(i);
      NT[i][idx + sz] = data(i + sz);
    }

  //Fill in missing values between known data.
  for (int i = 1; i < sz; ++i)
    for (int idx = sz - i; idx < sz;  ++idx)
      NT[i][idx] = NT[i-1][idx+1] - NT[i-1][idx];

  //Fill in final part of a table
  for (int i = sz; i < 2*sz; ++i)
    for (int idx = 0; idx < 2*sz - i; ++idx)
      NT[i][idx] = NT[i-1][idx+1] - NT[i-1][idx];

  //Get coeffictions
  for(int i = 0; i < 2*sz; ++i)
    data(i)= NT[i][0];

  // Change basis
  for (int k = 2*sz-2; k >= 0; --k)
    for (int j = k; j < 2*sz-1; ++j)
      if (k < sz)
        data(j) += 0.5 * data(j+1);
      else
        data(j) -= 0.5 * data(j+1);
};

template<int m, class ViewHermiteData>
KOKKOS_INLINE_FUNCTION
void interpolate2D(ViewHermiteData view_hermite_data) {
  KOKKOS_ASSERT(view_hermite_data.rank == 2);
  KOKKOS_ASSERT(view_hermite_data.extent(0) == 2*m+3);
  KOKKOS_ASSERT(view_hermite_data.extent(1) == 2*m+3);

  for (int i = 0; i < 2*m+2; ++i) {
    auto dd1 = Kokkos::subview(view_hermite_data, Kokkos::ALL, i);
    interpolateInPlace1D<m>(dd1);
  }

  for (int i = 0; i < 2*m+2; ++i) {
    auto dd1 = Kokkos::subview(view_hermite_data, i, Kokkos::ALL);
    interpolateInPlace1D<m>(dd1);
  }

}

template<int m, class HermiteView>
void interpolate_grid (HermiteView hermite_data) {
  static_assert(HermiteView::rank == 5);

  using exec_space = typename HermiteView::execution_space;

  const size_t n0 = hermite_data.extent(2);
  const size_t n1 = hermite_data.extent(3);
  const size_t n2 = hermite_data.extent(4);

  Kokkos::parallel_for("interpolate",
  Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0,0,0},{n0, n1, n2}),
  KOKKOS_LAMBDA(int component, int i, int j) {
    auto sbv_hermite_data = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, component, i, j);
    interpolate2D<m>(sbv_hermite_data);
  });
}

template<int m, int swidth = 7>
struct Interpolator {
  template<class DataView, class HermiteView>
  void interpolate(const StructuredLocator& fd_locator,
                   const StructuredLocator& hermite_locator,
                   DataView data,
                   HermiteView hermite_data) const {

    const Real scaleR = hermite_locator.dR / fd_locator.dR;
    const Real scaleZ = hermite_locator.dZ / fd_locator.dZ;

    compute_derivatives_grid<m, swidth>(data, hermite_data, scaleR, scaleZ);
    interpolate_grid<m>(hermite_data);
  }


  template<int ncomp, class DataView, class HermiteView>
  void interpolateRange(const StructuredLocator& fd_locator,
                   const StructuredLocator& hermite_locator,
                   DataView data,
                   HermiteView hermite_data, int component0 = 0) const {
     KOKKOS_ASSERT(component0 >= 0);
     KOKKOS_ASSERT(component0 + ncomp <= data.extent_int(2));
     KOKKOS_ASSERT(component0 + ncomp <= hermite_data.extent_int(2));


		auto dd = Kokkos::subview(data, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(component0, component0 + ncomp));
		auto hh = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(component0, component0 + ncomp), Kokkos::ALL, Kokkos::ALL);

    interpolate(fd_locator, hermite_locator, dd, hh);
  }

  template<class DataView, class HermiteView>
  void interpolateComponent(const StructuredLocator& fd_locator,
                   const StructuredLocator& hermite_locator,
                   DataView data,
                   HermiteView hermite_data,
                   int component) const {

     interpolateRange<1>(fd_locator, hermite_locator, data, hermite_data, component);
  }



  template<class HermiteView>
  void cleanDivergence(const StructuredLocator& hermite_locator, HermiteView hermite_data, int component0 = 0, int nfields = 1, int component_stride = 3)
  {
    static_assert(HermiteView::rank == 5,
                  "cleanDivergence expects rank-5 view: (idR,idZ,di,iR,iZ)");
    KOKKOS_ASSERT(component0 >= 0);
    KOKKOS_ASSERT(component0 + (nfields - 1) * component_stride + 2 < hermite_data.extent_int(2));

    using scalar_t = typename HermiteView::non_const_value_type;

    // hermite_data(idR, idZ, di, iR, iZ)

    const int Pr   = hermite_data.extent_int(0);  // # idR coefficients
    const int Pz   = hermite_data.extent_int(1);  // # idZ coefficients (incl. constant term k=0)
    const int nR   = hermite_data.extent_int(3);  // # radial cells
    const int nZ   = hermite_data.extent_int(4);  // # axial cells

    const int iZ0 = nZ / 2;

    const scalar_t hZ_over_hR = static_cast<scalar_t>(hermite_locator.dZ / hermite_locator.dR);

    using exec_space = typename HermiteView::execution_space;
    using policy_t   = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    // One work-item per (iRcell, idR). Inside we do:
    //  (1) local fill of non-constant z-coeffs from RBR
    //  (2) O(nZ) marching integration for the constant term a0 to enforce continuity
    Kokkos::parallel_for("cleanDivergence", policy_t({0, 0, 0}, {nfields, nR, Pr}),
      KOKKOS_LAMBDA(const int ifield, const int iRcell, const int idR)
      {
        int base = component0 + ifield * component_stride;
        // Anchor: preserve the existing constant coefficient at the center plane
        const scalar_t a0_center = hermite_data(idR, 0, base + 2, iRcell, iZ0);

        // --- (1) Fill non-constant Z coefficients from RBR (and init a0 everywhere to anchor)
        // RBZ(idR,k) = - RBR(idR+1,k-1) * (hZ/hR) * (idR+1)/k   for k>=1
        const scalar_t scale = -hZ_over_hR * static_cast<scalar_t>(idR + 1);

        for (int iZcell = 0; iZcell < nZ; ++iZcell) {
          hermite_data(idR, 0, base + 2, iRcell, iZcell) = a0_center;

          for (int k = 1; k < Pz; ++k) {
            scalar_t val = scalar_t(0);

            // Bounds-checked so we never read past RBR extents
            if ((idR + 1) < Pr && (k - 1) < Pz) {
              val = hermite_data(idR + 1, k - 1, base + 0, iRcell, iZcell) * scale / static_cast<scalar_t>(k);
            }

            hermite_data(idR, k, base + 2, iRcell, iZcell) = val;
          }
        }

        // --- (2) March upward/downward to set a0 so that RBZ is continuous at cell interfaces

        // Center cell non-constant contributions at boundaries:
        scalar_t sum_plus0  = scalar_t(0);
        scalar_t sum_minus0 = scalar_t(0);
        eval_nonconst_at_half(hermite_data, idR, base + 2, iRcell, iZ0, Pz, sum_plus0, sum_minus0);

        // Center cell boundary values:
        scalar_t boundary_top    = a0_center + sum_plus0;   // at Δz = +0.5
        scalar_t boundary_bottom = a0_center + sum_minus0;  // at Δz = -0.5

        // Upward sweep: enforce bottom boundary match to previous top boundary
        scalar_t boundary = boundary_top;
        for (int iZcell = iZ0 + 1; iZcell < nZ; ++iZcell) {
          scalar_t sum_plus  = scalar_t(0);
          scalar_t sum_minus = scalar_t(0);
          eval_nonconst_at_half(hermite_data, idR, base + 2, iRcell, iZcell, Pz, sum_plus, sum_minus);

          // Want: a0 + sum_minus == boundary   (match at Δz = -0.5)
          const scalar_t a0 = boundary - sum_minus;
          hermite_data(idR, 0, base + 2, iRcell, iZcell) = a0;

          // Next boundary is this cell's top boundary (Δz = +0.5)
          boundary = a0 + sum_plus;
        }

        // Downward sweep: enforce top boundary match to previous bottom boundary
        boundary = boundary_bottom;
        for (int iZcell = iZ0; iZcell-- > 0; ) { // iZ0-1 ... 0 (safe even if iZ0==0)
          scalar_t sum_plus  = scalar_t(0);
          scalar_t sum_minus = scalar_t(0);
          eval_nonconst_at_half(hermite_data, idR, base + 2, iRcell, iZcell, Pz, sum_plus, sum_minus);

          // Want: a0 + sum_plus == boundary    (match at Δz = +0.5)
          const scalar_t a0 = boundary - sum_plus;
          hermite_data(idR, 0, base + 2, iRcell, iZcell) = a0;

          // Next boundary is this cell's bottom boundary (Δz = -0.5)
          boundary = a0 + sum_minus;
        }
      });
  };


  template<class HermiteView>
  void computeChi(const StructuredLocator& hermite_locator,
                   HermiteView hermite_data,
                   int component0,
                   int nfields,
                   int component_stride) {
    static_assert(HermiteView::rank == 5,
                  "computeFlux expects rank-5 view: (idR,idZ,di,iR,iZ)");

    KOKKOS_ASSERT(component0 >= 0);
    KOKKOS_ASSERT(component0 + (nfields - 1) * component_stride + 2 < hermite_data.extent_int(2));

    using scalar_t = typename HermiteView::non_const_value_type;

    // hermite_data(idR, idZ, di, iR, iZ)
    // idR - polynomial coefficient index in R
    // idZ - polynomial coefficient index in Z
    // di - field component index: 0 - RB_R, 1 - RB_phi, 2 - RB_Z
    // iR - cell index in R
    // iZ - cell index in Z

    const int Pr   = hermite_data.extent_int(0);  // # idR coefficients in RB_Z
    const int Pz   = hermite_data.extent_int(1);  // # idZ coefficients in RB_Z
    const int nR   = hermite_data.extent_int(3);  // # radial cells
    const int nZ   = hermite_data.extent_int(4);  // # axial cells


    const int iR0 = nR / 2;
    const int iZ0 = nZ / 2;

    const scalar_t hR_s = static_cast<scalar_t>(hermite_locator.dR);
    const scalar_t hZ_s = static_cast<scalar_t>(hermite_locator.dZ);
    const scalar_t half = scalar_t(0.5);
    const scalar_t minus_half = scalar_t(-0.5);

    using exec_space = typename HermiteView::execution_space;
    using policy_t   = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;
    using policy1D_t = Kokkos::RangePolicy<exec_space>;

    // compute Z integral and store it in psi coefficients. Thats is psi := - int_Zc^Z RB_R(R,Z') dZ'
    Kokkos::parallel_for("computeChi", policy_t({0, 0, 0}, {nfields, nR, Pr}),
      KOKKOS_LAMBDA(const int ifield, const int iRcell, const int idR)
      {
        int base = component0 + ifield * component_stride;
        for (int iZcell = 0; iZcell < nZ; ++iZcell) {
          for (int idZ = 0; idZ < Pz; ++idZ) {
            hermite_data(idR, idZ, base + 3, iRcell, iZcell) = scalar_t(0);
          }

          for (int idZ = 1; idZ < Pz; ++idZ) {
            scalar_t val = scalar_t(0);
            if (idR < Pr && (idZ - 1) < Pz) {
              val = -hZ_s * hermite_data(idR, idZ - 1, base + 1, iRcell, iZcell) /
                    static_cast<scalar_t>(idZ);
            }

            hermite_data(idR, idZ, base + 3, iRcell, iZcell) = val;
          }
        }

        // Anchor Zc is the lower edge of the central Z cell.
        scalar_t sum_plus = scalar_t(0);
        scalar_t sum_minus = scalar_t(0);
        scalar_t p_plus = half;
        scalar_t p_minus = minus_half;
        for (int idZ = 1; idZ < Pz; ++idZ) {
          const scalar_t ak = hermite_data(idR, idZ, base + 3, iRcell, iZ0);
          sum_plus += ak * p_plus;
          sum_minus += ak * p_minus;
          p_plus *= half;
          p_minus *= minus_half;
        }

        scalar_t a0 = -sum_minus;
        hermite_data(idR, 0, base + 3, iRcell, iZ0) = a0;
        scalar_t boundary = a0 + sum_plus;

        for (int iZcell = iZ0 + 1; iZcell < nZ; ++iZcell) {
          sum_plus = scalar_t(0);
          sum_minus = scalar_t(0);
          p_plus = half;
          p_minus = minus_half;
          for (int idZ = 1; idZ < Pz; ++idZ) {
            const scalar_t ak = hermite_data(idR, idZ, base + 3, iRcell, iZcell);
            sum_plus += ak * p_plus;
            sum_minus += ak * p_minus;
            p_plus *= half;
            p_minus *= minus_half;
          }

          a0 = boundary - sum_minus;
          hermite_data(idR, 0, base + 3, iRcell, iZcell) = a0;
          boundary = a0 + sum_plus;
        }

        boundary = scalar_t(0);
        for (int iZcell = iZ0; iZcell-- > 0; ) {
          sum_plus = scalar_t(0);
          sum_minus = scalar_t(0);
          p_plus = half;
          p_minus = minus_half;
          for (int idZ = 1; idZ < Pz; ++idZ) {
            const scalar_t ak = hermite_data(idR, idZ, base + 3, iRcell, iZcell);
            sum_plus += ak * p_plus;
            sum_minus += ak * p_minus;
            p_plus *= half;
            p_minus *= minus_half;
          }

          a0 = boundary - sum_plus;
          hermite_data(idR, 0, base + 3, iRcell, iZcell) = a0;
          boundary = a0 + sum_minus;
        }
      });
  }

  template<class HermiteView, class PsiView>
  void computeFlux(const StructuredLocator& hermite_locator,
                   HermiteView hermite_data,
                   PsiView psi_hermite_data, int component0 = 0)
  {
    static_assert(HermiteView::rank == 5,
                  "computeFlux expects rank-5 view: (idR,idZ,di,iR,iZ)");
    static_assert(PsiView::rank == 4,
                  "computeFlux expects rank-4 psi view: (idR,idZ,iR,iZ)");

    KOKKOS_ASSERT(component0 >= 0);
    KOKKOS_ASSERT(component0 + 2 < hermite_data.extent_int(2));

    using scalar_t = typename HermiteView::non_const_value_type;

    // hermite_data(idR, idZ, di, iR, iZ)
    // idR - polynomial coefficient index in R
    // idZ - polynomial coefficient index in Z
    // di - field component index: 0 - RB_R, 1 - RB_phi, 2 - RB_Z
    // iR - cell index in R
    // iZ - cell index in Z

    const int Pr   = hermite_data.extent_int(0);  // # idR coefficients in RB_Z
    const int Pz   = hermite_data.extent_int(1);  // # idZ coefficients in RB_Z
    const int nR   = hermite_data.extent_int(3);  // # radial cells
    const int nZ   = hermite_data.extent_int(4);  // # axial cells


    const int PpsiR = psi_hermite_data.extent_int(0);
    const int PpsiZ = psi_hermite_data.extent_int(1);

    const int iR0 = nR / 2;
    const int iZ0 = nZ / 2;

    const scalar_t hR_s = static_cast<scalar_t>(hermite_locator.dR);
    const scalar_t hZ_s = static_cast<scalar_t>(hermite_locator.dZ);
    const scalar_t half = scalar_t(0.5);
    const scalar_t minus_half = scalar_t(-0.5);

    using exec_space = typename HermiteView::execution_space;
    using policy_t   = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;
    using policy1D_t = Kokkos::RangePolicy<exec_space>;

    // compute Z integral and store it in psi coefficients. Thats is psi := - int_Zc^Z RB_R(R,Z') dZ'
    Kokkos::parallel_for("computeFlux_Z", policy_t({0, 0}, {nR, PpsiR}),
      KOKKOS_LAMBDA(const int iRcell, const int idR)
      {
        int ifield = 0, component_stride = 3;
        int base = component0 + ifield * component_stride;
        for (int iZcell = 0; iZcell < nZ; ++iZcell) {
          for (int idZ = 0; idZ < PpsiZ; ++idZ) {
            psi_hermite_data(idR, idZ, iRcell, iZcell) = scalar_t(0);
          }

          for (int idZ = 1; idZ < PpsiZ; ++idZ) {
            scalar_t val = scalar_t(0);
            if (idR < Pr && (idZ - 1) < Pz) {
              val = -hZ_s * hermite_data(idR, idZ - 1, base, iRcell, iZcell) /
                    static_cast<scalar_t>(idZ);
            }

            psi_hermite_data(idR, idZ, iRcell, iZcell) = val;
          }
        }

        // Anchor Zc is the lower edge of the central Z cell.
        scalar_t sum_plus = scalar_t(0);
        scalar_t sum_minus = scalar_t(0);
        scalar_t p_plus = half;
        scalar_t p_minus = minus_half;
        for (int idZ = 1; idZ < PpsiZ; ++idZ) {
          const scalar_t ak = psi_hermite_data(idR, idZ, iRcell, iZ0);
          sum_plus += ak * p_plus;
          sum_minus += ak * p_minus;
          p_plus *= half;
          p_minus *= minus_half;
        }

        scalar_t a0 = -sum_minus;
        psi_hermite_data(idR, 0, iRcell, iZ0) = a0;
        scalar_t boundary = a0 + sum_plus;

        for (int iZcell = iZ0 + 1; iZcell < nZ; ++iZcell) {
          sum_plus = scalar_t(0);
          sum_minus = scalar_t(0);
          p_plus = half;
          p_minus = minus_half;
          for (int idZ = 1; idZ < PpsiZ; ++idZ) {
            const scalar_t ak = psi_hermite_data(idR, idZ, iRcell, iZcell);
            sum_plus += ak * p_plus;
            sum_minus += ak * p_minus;
            p_plus *= half;
            p_minus *= minus_half;
          }

          a0 = boundary - sum_minus;
          psi_hermite_data(idR, 0, iRcell, iZcell) = a0;
          boundary = a0 + sum_plus;
        }

        boundary = scalar_t(0);
        for (int iZcell = iZ0; iZcell-- > 0; ) {
          sum_plus = scalar_t(0);
          sum_minus = scalar_t(0);
          p_plus = half;
          p_minus = minus_half;
          for (int idZ = 1; idZ < PpsiZ; ++idZ) {
            const scalar_t ak = psi_hermite_data(idR, idZ, iRcell, iZcell);
            sum_plus += ak * p_plus;
            sum_minus += ak * p_minus;
            p_plus *= half;
            p_minus *= minus_half;
          }

          a0 = boundary - sum_plus;
          psi_hermite_data(idR, 0, iRcell, iZcell) = a0;
          boundary = a0 + sum_minus;
        }
      });

    Kokkos::fence();

    // compute R integral and add to psi coefficients. That is psi += int_Rc RB_Z (R',Zc) dR'
    Kokkos::parallel_for("computeFlux_R", policy1D_t(0, nZ),
      KOKKOS_LAMBDA(const int iZcell)
      {
        int ifield = 0, component_stride = 3;
        int base = component0 + ifield * component_stride;
        for (int iRcell = 0; iRcell < nR; ++iRcell) {
          for (int idR = 1; idR < PpsiR; ++idR) {
            scalar_t z_anchor_coeff = scalar_t(0);
            if ((idR - 1) < Pr) {
              z_anchor_coeff = eval_z_at(hermite_data, idR - 1, base + 2, iRcell, iZ0, Pz, minus_half);
            }

            psi_hermite_data(idR, 0, iRcell, iZcell) +=
                hR_s * z_anchor_coeff / static_cast<scalar_t>(idR);
          }
        }

        // Anchor Rc is the lower edge of the central R cell.
        scalar_t sum_plus = scalar_t(0);
        scalar_t sum_minus = scalar_t(0);
        scalar_t p_plus = half;
        scalar_t p_minus = minus_half;
        for (int idR = 1; idR < PpsiR; ++idR) {
          scalar_t coeff = scalar_t(0);
          if ((idR - 1) < Pr) {
            coeff = hR_s * eval_z_at(hermite_data, idR - 1, base + 2, iR0, iZ0, Pz, minus_half) /
                    static_cast<scalar_t>(idR);
          }
          sum_plus += coeff * p_plus;
          sum_minus += coeff * p_minus;
          p_plus *= half;
          p_minus *= minus_half;
        }

        scalar_t a0 = -sum_minus;
        psi_hermite_data(0, 0, iR0, iZcell) += a0;
        scalar_t boundary = a0 + sum_plus;

        for (int iRcell = iR0 + 1; iRcell < nR; ++iRcell) {
          sum_plus = scalar_t(0);
          sum_minus = scalar_t(0);
          p_plus = half;
          p_minus = minus_half;
          for (int idR = 1; idR < PpsiR; ++idR) {
            scalar_t coeff = scalar_t(0);
            if ((idR - 1) < Pr) {
              coeff = hR_s * eval_z_at(hermite_data, idR - 1, base + 2, iRcell, iZ0, Pz, minus_half) /
                      static_cast<scalar_t>(idR);
            }
            sum_plus += coeff * p_plus;
            sum_minus += coeff * p_minus;
            p_plus *= half;
            p_minus *= minus_half;
          }

          a0 = boundary - sum_minus;
          psi_hermite_data(0, 0, iRcell, iZcell) += a0;
          boundary = a0 + sum_plus;
        }

        boundary = scalar_t(0);
        for (int iRcell = iR0; iRcell-- > 0; ) {
          sum_plus = scalar_t(0);
          sum_minus = scalar_t(0);
          p_plus = half;
          p_minus = minus_half;
          for (int idR = 1; idR < PpsiR; ++idR) {
            scalar_t coeff = scalar_t(0);
            if ((idR - 1) < Pr) {
              coeff = hR_s * eval_z_at(hermite_data, idR - 1, base + 2, iRcell, iZ0, Pz, minus_half) /
                      static_cast<scalar_t>(idR);
            }
            sum_plus += coeff * p_plus;
            sum_minus += coeff * p_minus;
            p_plus *= half;
            p_minus *= minus_half;
          }

          a0 = boundary - sum_plus;
          psi_hermite_data(0, 0, iRcell, iZcell) += a0;
          boundary = a0 + sum_minus;
        }
      });
  }

  template<class HermiteView>
  void computeFluxComponent(const StructuredLocator& hermite_locator,
                   HermiteView hermite_data,
                   int psi_component, int b_component0 = 0)
  {

    static_assert(HermiteView::rank == 5,
                  "computeFlux expects rank-5 view: (idR,idZ,di,iR,iZ)");

    KOKKOS_ASSERT(psi_component < hermite_data.extent_int(2));

    auto psi = Kokkos::subview(hermite_data,
                               Kokkos::ALL(), Kokkos::ALL(), psi_component,
                               Kokkos::ALL(), Kokkos::ALL());

    computeFlux(hermite_locator, hermite_data, psi, b_component0);
  }
};


