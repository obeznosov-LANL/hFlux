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
    // Smooth (C-infinity), phi-periodic perturbation streamfunction:
    //   psi = exp(-(x^2 + z^2) / (2 sigma^2)) * sin(n phi)
    // with x = R - R_a, z = Z. Unlike a polar-angle form, this has no
    // singularity at the magnetic axis (r = 0), so Hermite/Fourier
    // interpolation retains its full convergence order.
    const Real x = R - R_a;
    const Real z = Z;

    psi = 0.0;
    dpsi_dR = 0.0;
    dpsi_dZ = 0.0;
    dpsi_dphi = 0.0;

    if (perturb_n == 0) {
      return;
    }

    const Real sigma = 0.5;
    const Real inv_sigma2 = 1.0 / (sigma * sigma);
    const Real envelope = Kokkos::exp(-0.5 * (x * x + z * z) * inv_sigma2);
    const Real sin_nphi = Kokkos::sin(static_cast<Real>(perturb_n) * phi);
    const Real cos_nphi = Kokkos::cos(static_cast<Real>(perturb_n) * phi);

    psi = envelope * sin_nphi;
    dpsi_dR = -x * inv_sigma2 * envelope * sin_nphi;
    dpsi_dZ = -z * inv_sigma2 * envelope * sin_nphi;
    dpsi_dphi = static_cast<Real>(perturb_n) * envelope * cos_nphi;
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
    B[1] = (bphi_amp * dpsi_dZ) / R;
    B[2] = ((R - R_a) / q(R, Z) + perturb_amp * dpsi_dR -
            bphi_amp * dpsi_dphi / R) / R;

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
