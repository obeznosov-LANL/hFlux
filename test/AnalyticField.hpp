#pragma once
#include "hFlux/common.hpp"

// Selects the analytic perturbation streamfunction:
//   Smooth   - C-infinity, non-resonant bump used by the convergence tests
//              (fourier_test / hermite_test), preserves full interpolation
//              order.
//   Resonant - island-producing perturbation resonant with q = m/n; only
//              C^m smooth at the axis, used by the Poincare plot.
enum class PerturbationKind { Smooth, Resonant };

struct AnalyticField {
  const Real q0, q2, R_a, E_0;
  const Real perturb_amp, bphi_amp;
  const int perturb_m, perturb_n;
  const PerturbationKind perturb_kind;

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0):
    AnalyticField(q0, q2, R_a, E_0, 0.0, 0.0, 2, 1) {}

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0,
                const Real perturb_amp):
    AnalyticField(q0, q2, R_a, E_0, perturb_amp, perturb_amp, 2, 1) {}

  AnalyticField(const Real q0, const Real q2, const Real R_a, const Real E_0,
                const Real perturb_amp, const PerturbationKind kind):
    AnalyticField(q0, q2, R_a, E_0, perturb_amp, perturb_amp, 2, 1, kind) {}

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
                const int perturb_m, const int perturb_n,
                const PerturbationKind kind = PerturbationKind::Smooth):
    q0(q0), q2(q2), R_a(R_a), E_0(E_0),
    perturb_amp(perturb_amp), bphi_amp(bphi_amp),
    perturb_m(perturb_m), perturb_n(perturb_n), perturb_kind(kind) {}

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
    psi = 0.0;
    dpsi_dR = 0.0;
    dpsi_dZ = 0.0;
    dpsi_dphi = 0.0;

    if (perturb_n == 0) {
      return;
    }

    if (perturb_kind == PerturbationKind::Resonant) {
      perturbation_resonant(psi, dpsi_dR, dpsi_dZ, dpsi_dphi, R, Z, phi);
    } else {
      perturbation_smooth(psi, dpsi_dR, dpsi_dZ, dpsi_dphi, R, Z, phi);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void perturbation_smooth(Real& psi, Real& dpsi_dR, Real& dpsi_dZ,
                           Real& dpsi_dphi, const Real R,
                           const Real Z, const Real phi) const {
    // Smooth (C-infinity), phi-periodic perturbation streamfunction:
    //   psi = exp(-(x^2 + z^2) / (2 sigma^2)) * sin(n phi)
    // with x = R - R_a, z = Z. Unlike a polar-angle form, this has no
    // singularity at the magnetic axis (r = 0), so Hermite/Fourier
    // interpolation retains its full convergence order.
    const Real x = R - R_a;
    const Real z = Z;

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
  void perturbation_resonant(Real& psi, Real& dpsi_dR, Real& dpsi_dZ,
                             Real& dpsi_dphi, const Real R,
                             const Real Z, const Real phi) const {
    // Resonant, phi-periodic perturbation streamfunction:
    //   psi = env(r) * sin(n phi - m theta)
    // with poloidal polar coordinates about the magnetic axis (R_a, 0):
    //   x = R - R_a, z = Z, r = sqrt(x^2 + z^2), theta = atan2(z, x).
    // The envelope is a Gaussian bump centred on the resonant surface where
    // q(r) = m / n, i.e. r_mn^2 = (m/n - q0) / q2, and vanishes (times r^m)
    // at the axis so psi stays smooth there. A perturbation resonant with the
    // rotational transform opens magnetic islands of poloidal mode number m.
    // Only C^m smooth at the axis, so it degrades interpolation convergence:
    // reserved for the Poincare plot, not the convergence tests.
    const Real x = R - R_a;
    const Real z = Z;

    const Real r2 = x * x + z * z;
    const Real r = Kokkos::sqrt(r2);

    // Guard the coordinate singularity at the axis; the envelope is ~0 there.
    const Real r_eps = 1e-8;
    if (r < r_eps) {
      return;
    }

    const Real m = static_cast<Real>(perturb_m);
    const Real n = static_cast<Real>(perturb_n);

    // Resonant minor radius where q(r) = m / n.
    const Real r_mn2 = (m / n - q0) / q2;
    const Real r_mn = (r_mn2 > 0.0) ? Kokkos::sqrt(r_mn2) : 0.0;

    // Gaussian bump centred on the resonant surface, forced to vanish at the
    // axis via the r^m factor to keep psi smooth (C^m) at r = 0.
    const Real width = 0.05;
    const Real inv_w2 = 1.0 / (width * width);
    const Real bump = Kokkos::exp(-0.5 * (r - r_mn) * (r - r_mn) * inv_w2);
    const Real rpow = Kokkos::pow(r, m);
    const Real env = rpow * bump;

    // dEnv/dr = (m/r) * env - (r - r_mn) * inv_w2 * env
    const Real denv_dr = env * (m / r - (r - r_mn) * inv_w2);

    const Real theta = Kokkos::atan2(z, x);
    const Real a = n * phi - m * theta;
    const Real sin_a = Kokkos::sin(a);
    const Real cos_a = Kokkos::cos(a);

    // dr/dx = x/r, dr/dz = z/r; dtheta/dx = -z/r^2, dtheta/dz = x/r^2.
    // dpsi/dx = denv_dr * (x/r) * sin_a + env * cos_a * (-m) * dtheta/dx
    // dpsi/dz = denv_dr * (z/r) * sin_a + env * cos_a * (-m) * dtheta/dz
    psi = env * sin_a;
    dpsi_dR = denv_dr * (x / r) * sin_a + env * cos_a * (m * z / r2);
    dpsi_dZ = denv_dr * (z / r) * sin_a - env * cos_a * (m * x / r2);
    dpsi_dphi = env * n * cos_a;
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
    B[1] = R_a / R + (bphi_amp * dpsi_dZ) / R;
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
