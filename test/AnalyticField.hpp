#pragma once
#include "hFlux/common.hpp"

struct AnalyticField {
  const Real q0, q2, R_a, E_0;

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0):
    q0(q0), q2(q2), R_a(R_a), E_0(E_0) {}

  KOKKOS_INLINE_FUNCTION
  Real q(const Real &R, const Real &Z) const {
    return q0 + q2 * (R - R_a) * (R - R_a) + q2 * Z * Z;
  }
  KOKKOS_INLINE_FUNCTION
  Real dqR(const Real &R, const Real &Z) const { return 2.0 * q2 * (R - R_a); }
  KOKKOS_INLINE_FUNCTION
  Real dqZ(const Real &R, const Real &Z) const { return 2.0 * q2 * Z; }

  KOKKOS_INLINE_FUNCTION
  ErrorCode operator()(const Dim5 &X, const Real &t, Dim3 &B, Dim3 &curlB,
                       Dim3 &dBdR, Dim3 &dBdZ, Dim3 &E) const {
    const Real R = X[2];
    const Real Z = X[4];

    B[0] =  -Z / q(R, Z) / R;        // B_R
    B[1] = R_a / R;                 // B_phi
    B[2] = (R - R_a) / q(R, Z) / R; // B_Z

    dBdR[0] = -B[0] / R - B[0] * dqR(R, Z) / q(R, Z);
    dBdR[1] = -R_a / R / R;
    dBdR[2] = 1.0 / q(R, Z) / R - B[2] / R - B[2] * dqR(R,Z) / q(R, Z);

    dBdZ[0] = -1.0 / q(R, Z) / R - B[0] * dqZ(R, Z) / q(R, Z);
    dBdZ[1] = 0.0;
    dBdZ[2] = -B[2] * dqZ(R, Z) / q(R, Z);

    curlB[0] = 0.0;
    curlB[1] = dBdZ[0] - dBdR[2];
    curlB[2] = 0.0;

    E[0] = 0.0;
    E[1] = E_0 * R_a / R;
    E[2] = 0.0;

    return ErrorCode::Success;
  };

  KOKKOS_INLINE_FUNCTION
  Real Psi(const Dim5 &X) const {
    const Real R = X[2];
    const Real Z = X[4];
    return log(q(R, Z)) / q2 * 0.5;
  }


  KOKKOS_INLINE_FUNCTION
  ErrorCode checkWall(const Dim5 &X) const { return ErrorCode::Success; }
};
