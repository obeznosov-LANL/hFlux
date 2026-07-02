#pragma once
#include "hFlux/common.hpp"

struct AnalyticField {
  const Real q0, q2, R_a, E_0;
  const Real perturb_amp, bphi_amp;
  const int perturb_m, perturb_n;

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0):
    AnalyticField(q0, q2, R_a, E_0, 0.0, 0.0, 2, 1) {}

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0,
                const Real perturb_amp):
    AnalyticField(q0, q2, R_a, E_0, perturb_amp, perturb_amp, 2, 1) {}

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0,
                const Real perturb_amp, const int perturb_m,
                const int perturb_n):
    AnalyticField(q0, q2, R_a, E_0, perturb_amp, perturb_amp,
                  perturb_m, perturb_n) {}

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0,
                const Real perturb_amp, const Real bphi_amp):
    AnalyticField(q0, q2, R_a, E_0, perturb_amp, bphi_amp, 2, 1) {}

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0,
                const Real perturb_amp, const Real bphi_amp,
                const int perturb_m, const int perturb_n):
    q0(q0), q2(q2), R_a(R_a), E_0(E_0),
    perturb_amp(perturb_amp), bphi_amp(bphi_amp),
    perturb_m(perturb_m), perturb_n(perturb_n) {}

  KOKKOS_INLINE_FUNCTION
  Real q(const Real &R, const Real &Z) const {
    return q0 + q2 * (R - R_a) * (R - R_a) + q2 * Z * Z;
  }
  KOKKOS_INLINE_FUNCTION
  Real dqR(const Real &R, const Real &Z) const { return 2.0 * q2 * (R - R_a); }
  KOKKOS_INLINE_FUNCTION
  Real dqZ(const Real &R, const Real &Z) const { return 2.0 * q2 * Z; }

  KOKKOS_INLINE_FUNCTION
  ErrorCode eval(Dim3 &B, const Real R, const Real Z) const {
    B[0] =  -Z / q(R, Z) / R;        // B_R
    B[1] = R_a / R;                 // B_phi
    B[2] = (R - R_a) / q(R, Z) / R; // B_Z

    return ErrorCode::Success;
  }

  KOKKOS_INLINE_FUNCTION
  void perturbation_derivatives(Real& psi, Real& dpsi_dR, Real& dpsi_dZ,
                                Real& dpsi_dphi, const Real R,
                                const Real Z, const Real phi) const {
    const Real x = R - R_a;
    const Real z = Z;
    const Real r2 = x * x + z * z;

    psi = 0.0;
    dpsi_dR = 0.0;
    dpsi_dZ = 0.0;
    dpsi_dphi = 0.0;

    if (r2 == 0.0 || perturb_n == 0) {
      return;
    }

    const Real r = Kokkos::sqrt(r2);
    const Real theta = Kokkos::atan2(z, x);
    const Real q_mn = static_cast<Real>(perturb_m) / static_cast<Real>(perturb_n);
    const Real alpha = static_cast<Real>(perturb_n) * phi -
                       static_cast<Real>(perturb_m) * theta;
    const Real f = Kokkos::exp(-(r - q_mn) * (r - q_mn));
    const Real df_dr = -2.0 * (r - q_mn) * f;
    const Real sin_alpha = Kokkos::sin(alpha);
    const Real cos_alpha = Kokkos::cos(alpha);

    const Real dr_dR = x / r;
    const Real dr_dZ = z / r;
    const Real dalpha_dR = static_cast<Real>(perturb_m) * z / r2;
    const Real dalpha_dZ = -static_cast<Real>(perturb_m) * x / r2;

    psi = f * sin_alpha;
    dpsi_dR = df_dr * dr_dR * sin_alpha + f * cos_alpha * dalpha_dR;
    dpsi_dZ = df_dr * dr_dZ * sin_alpha + f * cos_alpha * dalpha_dZ;
    dpsi_dphi = static_cast<Real>(perturb_n) * f * cos_alpha;
  }

  KOKKOS_INLINE_FUNCTION
  ErrorCode eval(Dim3 &B, const Real R, const Real Z, const Real phi) const {
    Real psi_perturb = 0.0;
    Real dpsi_dR = 0.0;
    Real dpsi_dZ = 0.0;
    Real dpsi_dphi = 0.0;
    perturbation_derivatives(psi_perturb, dpsi_dR, dpsi_dZ, dpsi_dphi,
                             R, Z, phi);

    B[0] = (-Z / q(R, Z) - perturb_amp * dpsi_dZ) / R;
    B[1] = (R_a + bphi_amp * dpsi_dZ) / R;
    B[2] = ((R - R_a) / q(R, Z) + perturb_amp * dpsi_dR -
            bphi_amp * dpsi_dphi) / R;

    return ErrorCode::Success;
  }

  KOKKOS_INLINE_FUNCTION
  ErrorCode eval_derivatives(Dim3 &dBdR, Dim3 &dBdZ, const Real R, const Real Z) const {
    Dim3 B = {};
    eval(B, R, Z);

    dBdR[0] = -B[0] / R - B[0] * dqR(R, Z) / q(R, Z);
    dBdR[1] = -R_a / R / R;
    dBdR[2] = 1.0 / q(R, Z) / R - B[2] / R - B[2] * dqR(R,Z) / q(R, Z);

    dBdZ[0] = -1.0 / q(R, Z) / R - B[0] * dqZ(R, Z) / q(R, Z);
    dBdZ[1] = 0.0;
    dBdZ[2] = -B[2] * dqZ(R, Z) / q(R, Z);

    return ErrorCode::Success;
  }

  KOKKOS_INLINE_FUNCTION
  ErrorCode eval_curl(Dim3 &curl, const Real R, const Real Z) const {

    Dim3 dBdR = {}, dBdZ = {};
    eval_derivatives(dBdR, dBdZ, R, Z);

    curl[0] = 0.0;
    curl[1] = dBdZ[0] - dBdR[2];
    curl[2] = 0.0;

    return ErrorCode::Success;
  }

  KOKKOS_INLINE_FUNCTION
  ErrorCode eval_E(Dim3 &E, const Real R, const Real Z) const {

    E[0] = 0.0;
    E[1] = E_0 * R_a / R;
    E[2] = 0.0;

    return ErrorCode::Success;
  };

  KOKKOS_INLINE_FUNCTION
  Real Psi(const Real R, const Real Z) const {
    return Kokkos::log(q(R, Z) / q0) / q2 * 0.5;
  }

  KOKKOS_INLINE_FUNCTION
  Real Psi(const Real R, const Real Z, const Real phi) const {
    Real psi_perturb = 0.0;
    Real dpsi_dR = 0.0;
    Real dpsi_dZ = 0.0;
    Real dpsi_dphi = 0.0;
    perturbation_derivatives(psi_perturb, dpsi_dR, dpsi_dZ, dpsi_dphi,
                             R, Z, phi);
    return Psi(R, Z) + perturb_amp * psi_perturb;
  }

  KOKKOS_INLINE_FUNCTION
  ErrorCode checkWall(const Dim5 &X) const { return ErrorCode::Success; }
};
