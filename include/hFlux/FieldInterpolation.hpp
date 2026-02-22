#pragma once
#include <Kokkos_Core.hpp>
#include <cmath>
#include "FiniteDifferenceWeights.hpp"
#include "common.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;

// hermite_data index convention (8D, LayoutLeft):
//   (idR, idZ, di, fi, k, ti, iR, iZ)
//   idR  - R monomial coefficient order
//   idZ  - Z monomial coefficient order
//   di   - dimension (0,1,2)
//   fi   - field variable index
//   k    - Fourier coefficient (phi)
//   ti   - time interpolation endpoint
//   iR   - R cell index
//   iZ   - Z cell index
//
// psi_hermite_data index convention (6D, LayoutLeft):
//   (idR, idZ, k, ti, iR, iZ)

template<int m, class ViewType>
KOKKOS_INLINE_FUNCTION
void cleanDivergence(ViewType hermite_data, const double hR, const double hZ) {
  // hermite_data here is a 5D subview: (idR, idZ, di, iR, iZ)
  // This process will modify a F_Z component of the input field F to make sure the div(F) = 0 analytcally
  // Taking a center of integration to be in the middle of computational domain
  const int nR_hermite_data = hermite_data.extent(3);
  const int nZ_hermite_data = hermite_data.extent(4);
  int iZ0 = nZ_hermite_data / 2;

  auto RBR = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, 0, Kokkos::ALL, Kokkos::ALL);
  auto RBZ = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, 2, Kokkos::ALL, Kokkos::ALL);

  // Make RF_Z(R,Z) = RBZ(R,Z_0) - match at the center with the original interpolating polynomial
  for (int iR = 0; iR < nR_hermite_data; ++iR)
    for (int idR = 0; idR < 2*m+2; ++idR) {
      for (int iZ = 0; iZ < nZ_hermite_data; ++iZ) {
          RBZ(idR, 0, iR, iZ) = RBZ(idR, 0, iR, iZ0); // Constant coefficeint match
          for (int idZ = 1; idZ < 2*m+2; ++idZ)
            RBZ(idR, idZ, iR, iZ) = 0.0;  // Non-constant coefficients 0
        }
        // Fill RBZ coefficients with local - Int dBRdR dZ
        for (int iZ = 0; iZ < nZ_hermite_data; ++iZ)
          for (int idZ = 1; idZ < 2*m+3; ++idZ) {
            if(idR + 1 < 2*m+2)
              RBZ(idR, idZ, iR, iZ) = - RBR(idR + 1, idZ - 1, iR, iZ) * hZ * static_cast<Real>(idR + 1) / hR / static_cast<Real>(idZ);
          }
        // Constant part with area from center to edges in cell iZ0
        for (int iZ = iZ0+1; iZ < nZ_hermite_data; ++iZ) {
          Real II = 0.0; // Integral over the entire cell
          for (int idZ = 1; idZ < 2*m+3; ++idZ) {
            II += RBZ(idR, idZ, iR, iZ) * (std::pow(0.5,idZ) - std::pow(-0.5,idZ));
            RBZ(idR, 0, iR, iZ) += RBZ(idR, idZ, iR, iZ0) * std::pow( 0.5, idZ);
            RBZ(idR, 0, iR, iZ) -= RBZ(idR, idZ, iR, iZ)  * std::pow(-0.5, idZ);
          }
          // Carry out integral to the end of the domain ammending the constant coefficient in Taylor expantion
          for (int iiZ = iZ+1; iiZ < nZ_hermite_data; ++iiZ)
            RBZ(idR, 0, iR, iiZ) += II;
        }
        // Repeat line integration towards bottm
        for (int iZ = 0; iZ < iZ0; ++iZ) {
          Real II = 0.0;
          for (int idZ = 1; idZ < 2*m+3; ++idZ) {
            II += RBZ(idR, idZ, iR, iZ) * (std::pow(-0.5,idZ) - std::pow(0.5,idZ));
            RBZ(idR, 0, iR, iZ) += RBZ(idR, idZ, iR, iZ0) * std::pow(-0.5, idZ);
            RBZ(idR, 0, iR, iZ) -= RBZ(idR, idZ, iR, iZ)  * std::pow( 0.5, idZ);
          }
          for (int iiZ = 0; iiZ < iZ; ++iiZ)
            RBZ(idR, 0, iR, iiZ) += II;
        }
      }
}

template<int m, class ViewType, class PsiViewType>
void computeFlux(ViewType hermite_data, PsiViewType psi_hermite_data, const double hR, const double hZ) {

  int order = 2*m+2;
  int nR = hermite_data.extent(6);
  int nZ = hermite_data.extent(7);
  int nphi = hermite_data.extent(4);
  int nt = hermite_data.extent(5);

  // RBR: di=0, fi=0 -> 6D (idR, idZ, k, ti, iR, iZ)
  auto RBR = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, 0, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  // RBZ: di=2, fi=0, iZ=0 -> 5D (idR, idZ, k, ti, iR)
  auto RBZ = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, 2, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0);

  Kokkos::View<double******, Kokkos::LayoutLeft, ExecSpace> intZ_RBR("intZ_RBR", order+1, order+1, nphi, nt, nR, nZ);
  Kokkos::View<double****, Kokkos::LayoutLeft, ExecSpace> intR_RBZ("intR_RBZ", order+1, nphi, nt, nR);

  {
    using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>;
    Kokkos::parallel_for("IntegrateZ",
        Policy({0, 1, 0, 0, 0, 0}, {order, order+1, nphi, nt, nR, nZ}),
        KOKKOS_LAMBDA(int idR, int idZ, int k, int l, int i, int j) {
          intZ_RBR(idR, idZ, k, l, i, j) = RBR(idR, idZ - 1, k, l, i, j) * hZ / static_cast<Real>(idZ);
        });
  }

  {
    using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
    Kokkos::parallel_for("IntegrateR",
        Policy({1, 0, 0, 0}, {order+1, nphi, nt, nR}),
        KOKKOS_LAMBDA(int idR, int k, int l, int i) {
          intR_RBZ(idR, k, l, i) = 0.0;
          for (int idZ = 0; idZ < 2*m+2; ++idZ) {
            intR_RBZ(idR, k, l, i) += RBZ(idR-1, idZ, k, l, i) * hR / static_cast<Real>(idR) * Kokkos::pow(-0.5, idZ);
          }
        });
  }
  Kokkos::fence();

  Kokkos::View<double*****, Kokkos::LayoutLeft, ExecSpace> intZ_RBR_Cell("intZ_RBR_Cell", order+1, nphi, nt, nR, nZ);
  Kokkos::View<double***, Kokkos::LayoutLeft, ExecSpace> intR_RBZ_Cell("intR_RBZ_Cell", nphi, nt, nR);

  {
    using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>;
    Kokkos::parallel_for("IntegrateZ",
        Policy({0, 0, 0, 0, 0}, {order, nphi, nt, nR, nZ}),
        KOKKOS_LAMBDA(int idR, int k, int l, int i, int j) {
          intZ_RBR_Cell(idR, k, l, i, j) = 0.0;
          for (int idZ = 1; idZ < 2*m+3; idZ += 2)
            intZ_RBR_Cell(idR, k, l, i, j) += intZ_RBR(idR, idZ, k, l, i, j) * 2.0 * Kokkos::pow(0.5, idZ);
          intZ_RBR(idR, 0, k, l, i, j) = 0.0;
          for (int idZ = 1; idZ < 2*m+3; ++idZ)
            intZ_RBR(idR, 0, k, l, i, j) -= intZ_RBR(idR, idZ, k, l, i, j) * Kokkos::pow(-0.5, idZ);
        });
  }

  {
    using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
    Kokkos::parallel_for("IntegrateR",
        Policy({0, 0, 0}, {nphi, nt, nR}),
        KOKKOS_LAMBDA(int k, int l, int i) {
          intR_RBZ_Cell(k, l, i) = 0.0;
          for (int idR = 1; idR < 2*m+3; idR += 2) {
            intR_RBZ_Cell(k, l, i) += intR_RBZ(idR, k, l, i) * 2.0 * Kokkos::pow(0.5, idR);
          }
          intR_RBZ(0, k, l, i) = 0.0;
          for (int idR = 1; idR < 2*m+3; ++idR) {
            intR_RBZ(0, k, l, i) -= intR_RBZ(idR, k, l, i) * Kokkos::pow(-0.5, idR);
          }
        });
  }

  Kokkos::fence();

  {
    using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>;
    Kokkos::parallel_for("Compute psi",
        Policy({0, 0, 0, 0, 0, 0}, {order+1, order+1, nphi, nt, nR, nZ}),
        KOKKOS_LAMBDA(int idR, int idZ, int k, int l, int i, int j) {
          psi_hermite_data(idR,idZ,k,l,i,j) = - intZ_RBR(idR, idZ, k, l, i, j);
          if (idZ == 0) {
            psi_hermite_data(idR,0,k,l,i,j) += intR_RBZ(idR, k, l, i);
            for (int jj = 0; jj < j; ++jj)
              psi_hermite_data(idR, 0, k, l, i, j) -= intZ_RBR_Cell(idR, k, l, i, jj);
            if (idR == 0)
              for (int ii = 0; ii < i; ++ii)
                psi_hermite_data(0, 0, k, l, i, j) += intR_RBZ_Cell(k, l, ii);
          }
        });
  }

  Kokkos::fence();
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
  assert(view_hermite_data.extent(0) == 2*m+2);
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

// Helper: create subview with ranges on first 2 dims, scalars from indices[] for the rest
template<int m, class ViewType, std::size_t... Is>
KOKKOS_INLINE_FUNCTION
auto subview_range2(ViewType v, int r0, int r1, const int* indices, std::index_sequence<Is...>) {
    return Kokkos::subview(v,
        Kokkos::make_pair(r0, r0 + m + 1),
        Kokkos::make_pair(r1, r1 + m + 1),
        indices[Is + 2]...);
}

template<int m, int swidth = 7, class DataViewType, class HermiteViewType>
void compute_derivatives(DataViewType data, HermiteViewType hermite_data,
                         const Real ratioR, const Real ratioZ) {
    assert(ratioR > 1.0);
    assert(ratioZ > 1.0);

    using exec_space = typename HermiteViewType::execution_space;
    static_assert(Kokkos::SpaceAccessibility<exec_space, typename DataViewType::memory_space>::accessible,
                  "Data view must be accessible from hermite_data's execution space");

    // Iterate over all dimensions except idR (0) and idZ (1)
    int total = 1;
    for (int d = 2; d < HermiteViewType::rank; ++d)
      total *= hermite_data.extent(d);

    Kokkos::parallel_for("compute_derivatives",
    Kokkos::RangePolicy<exec_space>(0, total),
    KOKKOS_LAMBDA(int flat) {
      int indices[HermiteViewType::rank];
      int remainder = flat;
      for (int d = 2; d < HermiteViewType::rank; ++d) {
        indices[d] = remainder % hermite_data.extent(d);
        remainder /= hermite_data.extent(d);
      }

      // hermite: (idR, idZ, di, fi, k, ti, iR, iZ)
      // iR and iZ are the last two indices
      constexpr int rank = HermiteViewType::rank;
      const int i = indices[rank - 2];
      const int j = indices[rank - 1];

      for (int offx = 0; offx < 2; ++offx) {
        for (int offy = 0; offy < 2; ++offy) {
          int ii = (i + offx) * (swidth-1);
          int jj = (j + offy) * (swidth-1);
          int idx = (m+1) * offx;
          int idy = (m+1) * offy;

          auto sbv_data = Kokkos::subview(data, Kokkos::make_pair(ii, ii + swidth),
                                                Kokkos::make_pair(jj, jj + swidth),
                                                indices[3], indices[2], indices[4], indices[5]);
          auto sbv_hermite_data = subview_range2<m>(hermite_data, idx, idy, indices,
                                                    std::make_index_sequence<rank - 2>{});
          computeDerivativesStencil<m, swidth>(sbv_data, sbv_hermite_data, ratioR, ratioZ);
        }
      }
    });
}

template<int m, int swidth = 7>
struct FieldInterpolation {
  const int nR_data, nZ_data;
  const int nfields;
  const int ndims = 3;
  const int nphi_data;
  const int nt;

  const Real R0, Z0;
  const Real dR, dZ;

  const int nR_hermite_data, nZ_hermite_data;
  const Real hR0, hZ0;
  const Real hR, hZ;

  Kokkos::View<Real******, Kokkos::LayoutLeft, ExecSpace> data;
  Kokkos::View<Real********, Kokkos::LayoutLeft, ExecSpace> hermite_data;


  public:

  FieldInterpolation(const int nR_data, const int nZ_data, const int nfields, const int nphi_data, const int nt,
                     const Real R0, const Real Z0, const Real dR, const Real dZ) :
    nR_data(nR_data), nZ_data(nZ_data), nfields(nfields), nphi_data(nphi_data), nt(nt),
    R0(R0), Z0(Z0), dR(dR), dZ(dZ),
    nR_hermite_data((nR_data-1) / (swidth-1) - 1), nZ_hermite_data((nZ_data-1) / (swidth-1) - 1),
    hR0(R0 + (swidth-1)/2*dR), hZ0(Z0 + (swidth-1)/2 *dZ),
    hR(dR * (swidth - 1)), hZ(dZ * (swidth - 1)),
    data("data", nR_data, nZ_data, nfields, ndims, nphi_data, nt),
    hermite_data("hermite_data", 2*m+2, 2*m+3, ndims, nfields, nphi_data, nt, nR_hermite_data, nZ_hermite_data) {
      std::printf("Initialized field interpolation,\n hR0 = %le, hZ0 = %le\n hR = %le, hZ = %le\n nR = %d, nZ = %d\n",
          hR0, hZ0, hR, hZ, nR_hermite_data, nZ_hermite_data);
  };

  void interpolate() {
    compute_derivatives<m, swidth>(data, hermite_data, hR / dR, hZ / dZ);

    auto hermite_data_ = hermite_data;

    Kokkos::parallel_for("interpolate",
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0,0,0,0,0,0}, {nR_hermite_data,nZ_hermite_data, nfields,ndims,nphi_data,nt}),
    KOKKOS_LAMBDA(int i, int j, int fi, int di, int k, int ti){
      auto sbv_hermite_data = Kokkos::subview(hermite_data_, Kokkos::ALL, Kokkos::ALL, di, fi, k, ti, i, j);
      interpolate2D<m>(sbv_hermite_data);
    });

    Real hR_ = hR, hZ_ = hZ;

    Kokkos::parallel_for("cleandiv",
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0}, {nphi_data,nt}),
    KOKKOS_LAMBDA(int k, int ti){
      auto sbv_hermite_data = Kokkos::subview(hermite_data_, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, k, ti, Kokkos::ALL, Kokkos::ALL);
      cleanDivergence<m>(sbv_hermite_data, hR_, hZ_);
    });
  }


  template<class ViewVals>
  KOKKOS_INLINE_FUNCTION
  ErrorCode operator()(ViewVals vals, Dim5 X) const
  {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    KOKKOS_ASSERT(hermite_data.extent(6) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(7) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ii, jj);

    for (int fi = 0; fi < nfields; ++fi)
      for (int di = 0; di < ndims; ++di)
        for (int k = 0; k < nphi_data; ++k)
          for (int ti = 0; ti < nt; ++ti) {
            vals(fi, di, k, ti) = 0.0;
            Real sclr = 1.0;
            for (int i = 0; i < sbv.extent(0); ++i) {
                Real sclz = 1.0;
                for (int j = 0; j < sbv.extent(1); ++j) {
                    Real mon = sclr * sclz;
                    vals(fi, di, k, ti) += mon * sbv(i, j, di, fi, k, ti);
                    sclz *= z;
                }
                sclr *= r;
            }
          }

    return ErrorCode::Success;
  }


  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  ErrorCode evalB(Dim3& B, Dim5 X, Real t, ViewType hermite_data) const {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    KOKKOS_ASSERT(hermite_data.extent(6) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(7) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, 0, Kokkos::ALL, ii, jj);

    for (int di = 0; di < ndims; ++di) {
      B[di] = 0.0;
      Real sclr = 1.0;
      for (int i = 0; i < sbv.extent(0); ++i) {
        Real sclz = 1.0;
        for (int j = 0; j < sbv.extent(1); ++j) {
            Real mon = sclr * sclz;
            B[di] += mon * (sbv(i, j, di, 0) * (1.0 - t) + sbv(i, j, di, 1) * t);
            sclz *= z;
        }
        sclr *= r;
      }
      B[di] /= X[2];
    }

    return ErrorCode::Success;
  }

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  ErrorCode evalB(Dim3& B, Dim5 X, ViewType hermite_data) const
  {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    if (hermite_data.extent(6) <= ii || ii < 0) return ErrorCode::WallImpact;
    if (hermite_data.extent(7) <= jj || jj < 0) return ErrorCode::WallImpact;

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, 0, 0, ii, jj);

    for (int di = 0; di < ndims; ++di) {
      B[di] = 0.0;
      Real sclr = 1.0;
      for (int i = 0; i < sbv.extent(0); ++i) {
        Real sclz = 1.0;
        for (int j = 0; j < sbv.extent(1); ++j) {
            Real mon = sclr * sclz;
            B[di] += mon * sbv(i, j, di);
            sclz *= z;
        }
        sclr *= r;
      }
      B[di] /= X[2];
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
    KOKKOS_ASSERT(hermite_data.extent(4) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(5) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, 0, 0, ii, jj);

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


  decltype(auto) getDataRef() & {
    return data;
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
          Dim3 B;
          ErrorCode status = evalB(B, X0, hermite_data);
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
