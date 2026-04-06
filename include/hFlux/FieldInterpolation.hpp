#pragma once
#include <Kokkos_Core.hpp>
#include <cmath>
#include "FiniteDifferenceWeights.hpp"
#include "common.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;

// hermite_data index convention (5D, LayoutRight):
//   (idR, idZ, di, iR, iZ)
//   idR  - R monomial coefficient order
//   idZ  - Z monomial coefficient order
//   di   - dimension (0,1,2)
//   iR   - R cell index
//   iZ   - Z cell index
//
// psi_hermite_data index convention (5D, LayoutRight):
//   (idR, idZ, di, iR, iZ)


template<int m, class HermiteViewType, class PsiViewType>
void computeFlux(HermiteViewType hermite_data,
                 PsiViewType psi_hermite_data,
                 const double hR,
                 const double hZ)
{
}

template<int m, class T>
KOKKOS_INLINE_FUNCTION
void interpolateInPlace1D(T& data) {
  assert(data.rank == 1);

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
}

template<int m, class ViewHermiteDataType>
KOKKOS_INLINE_FUNCTION
void interpolate2D(ViewHermiteDataType view_hermite_data) {
  assert(view_hermite_data.rank == 2);
  assert(view_hermite_data.extent(0) == 2*m+3);
  assert(view_hermite_data.extent(1) == 2*m+3);

  for (int i = 0; i < 2*m+2; ++i) {
    auto dd1 = Kokkos::subview(view_hermite_data, Kokkos::ALL, i);
    interpolateInPlace1D<m>(dd1);
  }

  for (int i = 0; i < 2*m+2; ++i) {
    auto dd1 = Kokkos::subview(view_hermite_data, i, Kokkos::ALL);
    interpolateInPlace1D<m>(dd1);
  }

}


template<int m, int swidth, class ViewDataType, class ViewHermiteDataType>
KOKKOS_INLINE_FUNCTION
void computeDerivativesStencil(ViewDataType view_data, ViewHermiteDataType view_hermite_data, const Real ratioR, const Real ratioZ) {
  assert(view_data.rank == 2);
  assert(view_data.extent(0) == swidth);
  assert(view_data.extent(1) == swidth);

  assert(view_hermite_data.rank == 2);
  assert(view_hermite_data.extent(0) == m+1);
  assert(view_hermite_data.extent(1) == m+1);

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
}


// Computes the finite difference derivates on point data collocated on some grid
// Stores the scaled derivatives (scaled) in hermite data, ready to be interpolated



template<int m = 2, int swidth = 7, class DataViewType, class HermiteViewType>
void compute_derivatives_grid (DataViewType data, HermiteViewType hermite_data,
                         const Real ratioR, const Real ratioZ) {

    static_assert(DataViewType::rank == 2);
    static_assert(HermiteViewType::rank == 4);

    assert(ratioR > 1.0);
    assert(ratioZ > 1.0);

    using exec_space = typename HermiteViewType::execution_space;

    const size_t n1 = hermite_data.extent(2);
    const size_t n2 = hermite_data.extent(3);

    Kokkos::deep_copy(hermite_data, 0.0);

    Kokkos::parallel_for("compute_derivatives",
    Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0,0},{n1, n2}),
    KOKKOS_LAMBDA(int i, int j) {
      for (int offx = 0; offx < 2; ++offx) {
        for (int offy = 0; offy < 2; ++offy) {
          int ii = (i + offx) * (swidth-1);
          int jj = (j + offy) * (swidth-1);
          int idx = (m+1) * offx;
          int idy = (m+1) * offy;

          auto data_stencil = Kokkos::subview(data,
              Kokkos::make_pair(ii, ii + swidth), Kokkos::make_pair(jj, jj + swidth));
          auto hermite_data_cell = Kokkos::subview(hermite_data,
              Kokkos::make_pair(idx, idx + m+1), Kokkos::make_pair(idy, idy + m+1), i, j);

          computeDerivativesStencil<m, swidth>(data_stencil, hermite_data_cell, ratioR, ratioZ);
        }
      }
    });
    Kokkos::fence();
}


template<int m = 2, class HermiteViewType>
void interpolate_grid (HermiteViewType hermite_data) {
  static_assert(HermiteViewType::rank == 4);

  using exec_space = typename HermiteViewType::execution_space;

  const size_t n1 = hermite_data.extent(2);
  const size_t n2 = hermite_data.extent(3);

  Kokkos::parallel_for("interpolate",
  Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0,0},{n1, n2}),
  KOKKOS_LAMBDA(int i, int j) {
    auto sbv_hermite_data = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, i, j);
    interpolate2D<m>(sbv_hermite_data);
  });
}

namespace detail {

template <class RBZViewType, class Scalar>
KOKKOS_INLINE_FUNCTION
void eval_nonconst_at_half(const RBZViewType& RBZ,
                           const int idR,
                           const int iRcell,
                           const int iZcell,
                           const int Pz,
                           Scalar& sum_plus,
                           Scalar& sum_minus)
{
  // sum_plus  = Σ_{k>=1} a_k * (+0.5)^k
  // sum_minus = Σ_{k>=1} a_k * (-0.5)^k
  sum_plus  = Scalar(0);
  sum_minus = Scalar(0);

  Scalar p_plus  = Scalar(0.5);   // (+0.5)^1
  Scalar p_minus = Scalar(-0.5);  // (-0.5)^1
  for (int k = 1; k < Pz; ++k) {
    const Scalar ak = RBZ(idR, k, iRcell, iZcell);
    sum_plus  += ak * p_plus;
    sum_minus += ak * p_minus;
    p_plus  *= Scalar(0.5);
    p_minus *= Scalar(-0.5);
  }
}

} // namespace detail


template<int m, class HermiteViewType>
void cleanDivergence(HermiteViewType hermite_data, const double hR, const double hZ)
{
  static_assert(HermiteViewType::rank == 5,
                "cleanDivergence expects rank-5 view: (idR,idZ,di,iR,iZ)");

  using scalar_t = typename HermiteViewType::non_const_value_type;

  // hermite_data(idR, idZ, di, iR, iZ)
  // Subviews become rank-4: (idR, idZ, iR, iZ)
  auto RBR = Kokkos::subview(hermite_data,
                             Kokkos::ALL(), Kokkos::ALL(), 0,
                             Kokkos::ALL(), Kokkos::ALL());
  auto RBZ = Kokkos::subview(hermite_data,
                             Kokkos::ALL(), Kokkos::ALL(), 2,
                             Kokkos::ALL(), Kokkos::ALL());

  const int Pr   = RBZ.extent_int(0);  // # idR coefficients
  const int Pz   = RBZ.extent_int(1);  // # idZ coefficients (incl. constant term k=0)
  const int nR   = RBZ.extent_int(2);  // # radial cells
  const int nZ   = RBZ.extent_int(3);  // # axial cells

  const int PrBR = RBR.extent_int(0);
  const int PzBR = RBR.extent_int(1);

  const int iZ0 = nZ / 2;             // same choice as your original code

#ifndef NDEBUG
  if (hR == 0.0) {
    Kokkos::abort("cleanDivergence: hR must be nonzero.");
  }
  if (nR <= 0 || nZ <= 0 || Pr <= 0 || Pz <= 0) {
    Kokkos::abort("cleanDivergence: empty extents.");
  }
  if (RBR.extent_int(2) != nR || RBR.extent_int(3) != nZ) {
    Kokkos::abort("cleanDivergence: RBR/RBZ iR/iZ extents mismatch.");
  }
#endif

  const scalar_t hZ_over_hR = static_cast<scalar_t>(hZ / hR);

  using exec_space = typename HermiteViewType::execution_space;
  using policy_t   = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;

  // One work-item per (iRcell, idR). Inside we do:
  //  (1) local fill of non-constant z-coeffs from RBR
  //  (2) O(nZ) marching integration for the constant term a0 to enforce continuity
  Kokkos::parallel_for("cleanDivergence", policy_t({0, 0}, {nR, Pr}),
    KOKKOS_LAMBDA(const int iRcell, const int idR)
    {
      // Anchor: preserve the existing constant coefficient at the center plane
      const scalar_t a0_center = RBZ(idR, 0, iRcell, iZ0);

      // --- (1) Fill non-constant Z coefficients from RBR (and init a0 everywhere to anchor)
      // RBZ(idR,k) = - RBR(idR+1,k-1) * (hZ/hR) * (idR+1)/k   for k>=1
      const scalar_t scale = -hZ_over_hR * static_cast<scalar_t>(idR + 1);

      for (int iZcell = 0; iZcell < nZ; ++iZcell) {
        RBZ(idR, 0, iRcell, iZcell) = a0_center;

        for (int k = 1; k < Pz; ++k) {
          scalar_t val = scalar_t(0);

          // Bounds-checked so we never read past RBR extents
          if ((idR + 1) < PrBR && (k - 1) < PzBR) {
            val = RBR(idR + 1, k - 1, iRcell, iZcell) * scale / static_cast<scalar_t>(k);
          }

          RBZ(idR, k, iRcell, iZcell) = val;
        }
      }

      // --- (2) March upward/downward to set a0 so that RBZ is continuous at cell interfaces

      // Center cell non-constant contributions at boundaries:
      scalar_t sum_plus0  = scalar_t(0);
      scalar_t sum_minus0 = scalar_t(0);
      detail::eval_nonconst_at_half(RBZ, idR, iRcell, iZ0, Pz, sum_plus0, sum_minus0);

      // Center cell boundary values:
      scalar_t boundary_top    = a0_center + sum_plus0;   // at Δz = +0.5
      scalar_t boundary_bottom = a0_center + sum_minus0;  // at Δz = -0.5

      // Upward sweep: enforce bottom boundary match to previous top boundary
      scalar_t boundary = boundary_top;
      for (int iZcell = iZ0 + 1; iZcell < nZ; ++iZcell) {
        scalar_t sum_plus  = scalar_t(0);
        scalar_t sum_minus = scalar_t(0);
        detail::eval_nonconst_at_half(RBZ, idR, iRcell, iZcell, Pz, sum_plus, sum_minus);

        // Want: a0 + sum_minus == boundary   (match at Δz = -0.5)
        const scalar_t a0 = boundary - sum_minus;
        RBZ(idR, 0, iRcell, iZcell) = a0;

        // Next boundary is this cell's top boundary (Δz = +0.5)
        boundary = a0 + sum_plus;
      }

      // Downward sweep: enforce top boundary match to previous bottom boundary
      boundary = boundary_bottom;
      for (int iZcell = iZ0; iZcell-- > 0; ) { // iZ0-1 ... 0 (safe even if iZ0==0)
        scalar_t sum_plus  = scalar_t(0);
        scalar_t sum_minus = scalar_t(0);
        detail::eval_nonconst_at_half(RBZ, idR, iRcell, iZcell, Pz, sum_plus, sum_minus);

        // Want: a0 + sum_plus == boundary    (match at Δz = +0.5)
        const scalar_t a0 = boundary - sum_plus;
        RBZ(idR, 0, iRcell, iZcell) = a0;

        // Next boundary is this cell's bottom boundary (Δz = -0.5)
        boundary = a0 + sum_minus;
      }
    });

   Kokkos::fence();
}


template<int m, int swidth = 7>
struct FieldInterpolation {
  const int nR_data, nZ_data;
  const int ndims = 3;

  const Real R0, Z0;
  const Real dR, dZ;

  const int nR_hermite_data, nZ_hermite_data;
  const Real hR0, hZ0;
  const Real hR, hZ;


  public:

  Kokkos::View<Real***, Kokkos::LayoutRight, ExecSpace> data;
  Kokkos::View<Real*****, Kokkos::LayoutLeft, ExecSpace> hermite_data;

  FieldInterpolation(const int nR_data, const int nZ_data,
                     const Real R0, const Real Z0, const Real dR, const Real dZ) :
    nR_data(nR_data), nZ_data(nZ_data),
    R0(R0), Z0(Z0), dR(dR), dZ(dZ),
    nR_hermite_data((nR_data-1) / (swidth-1) - 1), nZ_hermite_data((nZ_data-1) / (swidth-1) - 1),
    hR0(R0 + (swidth-1)/2*dR), hZ0(Z0 + (swidth-1)/2 *dZ),
    hR(dR * (swidth - 1)), hZ(dZ * (swidth - 1)),
    data("data", nR_data, nZ_data, ndims),
    hermite_data("hermite_data", 2*m+3, 2*m+3, ndims, nR_hermite_data, nZ_hermite_data) {
  };

  void interpolate() {
		 for (int i = 0; i < ndims; ++i) {
		   auto hh = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, i, Kokkos::ALL, Kokkos::ALL);
		   auto dd = Kokkos::subview(data, Kokkos::ALL, Kokkos::ALL, i);
       compute_derivatives_grid<m,swidth>(dd, hh, hR / dR, hZ / dZ);
       interpolate_grid<m>(hh);
		 }
     cleanDivergence<m>(hermite_data, hR, hZ);
  };


  template<class ViewVals>
  KOKKOS_INLINE_FUNCTION
  ErrorCode operator()(ViewVals vals, Dim5 X) const {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    KOKKOS_ASSERT(hermite_data.extent(2) == vals.extent(0));
    KOKKOS_ASSERT(hermite_data.extent(3) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(4) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ii, jj);

      for (int di = 0; di < ndims; ++di) {
        vals(di) = 0.0;
        Real sclr = 1.0;
        for (int i = 0; i < sbv.extent(0); ++i) {
          Real sclz = 1.0;
          for (int j = 0; j < sbv.extent(1); ++j) {
            Real mon = sclr * sclz;
            vals(di) += mon * sbv(i, j, di);
            sclz *= z;
        }
        sclr *= r;
			}
    }

    return ErrorCode::Success;
  }

  template<class HermiteViewType>
  KOKKOS_INLINE_FUNCTION
  ErrorCode eval_array(Dim3& vals, Dim5 X, HermiteViewType hermite_data) const
  {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    KOKKOS_ASSERT(hermite_data.extent(2) == 3);
    KOKKOS_ASSERT(hermite_data.extent(3) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(4) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ii, jj);

      for (int di = 0; di < ndims; ++di) {
        vals[di] = 0.0;
        Real sclr = 1.0;
        for (int i = 0; i < sbv.extent(0); ++i) {
          Real sclz = 1.0;
          for (int j = 0; j < sbv.extent(1); ++j) {
            Real mon = sclr * sclz;
            vals[di] += mon * sbv(i, j, di);
            sclz *= z;
        }
        sclr *= r;
			}
    }

    return ErrorCode::Success;
  }



  template<class PsiViewType>
  KOKKOS_INLINE_FUNCTION
  ErrorCode evalPsi(Real& val, Dim5 X, PsiViewType hermite_data) const {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    KOKKOS_ASSERT(hermite_data.extent(2) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(3) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL,  ii, jj);

    val = 0.0;
    Real sclr = 1.0;
    for (int i = 0; i < sbv.extent(0); ++i) {
      Real sclz = 1.0;
      for (int j = 0; j < sbv.extent(1); ++j) {
          Real mon = sclr * sclz;
          val += mon * sbv(i, j);
          sclz *= z;
      }
      sclr *= r;
    }

    return ErrorCode::Success;
  }


  std::array<Real, 4> getCorners() {
    return {hR0, hR0 + nR_hermite_data * hR, hZ0, hZ0 + nZ_hermite_data * hZ};
  }


  template<class ViewType, class PsiViewType>
  Real isd(Dim5& X0, ViewType hermite_data, PsiViewType psi_data, int sign) {
      assert(sign == -1 || sign == 1);

      // Declare Variables
      double tol = 1e-9; // tolerance for convergence
      int iter = 0;
      int max_iter = 100000; // maximum number of iterations


      // coefficients for gradient
      const Real alpha = 1.1; // expansion
      const Real beta = 0.5; // contraction
      Real ds = 0.5; // gradient variable
      Real grad, gradx, grady, coeff;
      Real dx, dy;
      Real last_fit, fit;
      Dim5 X;

      bool constraint = true;

      evalPsi(last_fit, X0, psi_data);
      fit = last_fit;

      //begin main loop
      for (iter = 0; iter < max_iter; iter++) {
          Dim3 B = {};
          ErrorCode status = eval_array(B, X0, hermite_data);
          assert(status == ErrorCode::Success);
          gradx =      B[2] * X0[2];
          grady =     -B[0] * X0[2];
          grad = std::sqrt(gradx * gradx + grady * grady);

          if (grad == 0){
              return fit;
          }

          coeff = ds / grad; // get cauchy coefficient

          X[2] = X0[2] + sign * coeff * gradx;
          X[4] = X0[4] + sign * coeff * grady;

          {
              //get new fitness
              status = evalPsi(fit, X, psi_data);
              assert(status == ErrorCode::Success);

              if (std::abs(fit-last_fit)<= tol){
                  return fit;
              }

              dx = X[2] - X0[2];
              dy = X[4] - X0[4];

              if (std::abs(dx) <= tol && std::abs(dy) <= tol){
                  return fit;
              }
          }

          // cauchy step was too big
          if (sign * (fit - last_fit) < 0 || !constraint) {

              ds *= beta;
          }
          else {

              ds *= alpha;
              last_fit = fit;
              X0[2] = X[2];
              X0[4] = X[4];
          }
      }

      if (iter == (max_iter -1)) {
        std::fprintf(stderr,"Solution did not converge quickly enough","");
      }
      else {
          return fit;
      }

      return fit; // return our best value i guess
  }

};
