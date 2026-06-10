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

#include "Evaluator.hpp"

template<class ViewType, class PsiViewType>
ErrorCode findMagneticAxis(Real& R_center, Real& Z_center,
              const ViewType hermite_data,
              const PsiViewType psi_data,
              const Evaluator& ev,
              const int sign,
              Real& fit) {
  assert(sign == -1 || sign == 1);

  const Real tol = 1e-9;
  const int max_iter = 100000;

  const Real alpha = 1.1;
  const Real beta = 0.5;
  Real ds = 0.5;

  Real last_fit = 0.0;
  ErrorCode status = ev.locator.checkBounds(R_center, Z_center);
  if (status != ErrorCode::Success) {
    return status;
  }
  ev.evalPsi(last_fit, R_center, Z_center, psi_data);

  fit = last_fit;

  for (int iter = 0; iter < max_iter; ++iter) {
    Dim3 B = {};
    status = ev.locator.checkBounds(R_center, Z_center);
    if (status != ErrorCode::Success) {
      return status;
    }

    ev.evalField(B, R_center, Z_center, hermite_data);

    const Real gradx = B[2] * R_center;
    const Real grady = -B[0] * R_center;
    const Real grad = std::sqrt(gradx * gradx + grady * grady);

    if (grad == 0.0) {
      fit = last_fit;
      return ErrorCode::Success;
    }

    const Real coeff = ds / grad;

    Real R = R_center + sign * coeff * gradx;
    Real Z = Z_center + sign * coeff * grady;

    ErrorCode status = ev.locator.checkBounds(R, Z);
    if (status == ErrorCode::OutOfBounds) {
      ds *= beta;
      continue;
    }

    ev.evalPsi(fit, R, Z, psi_data);

    const Real dfit = std::abs(fit - last_fit);
    const Real dx = R - R_center;
    const Real dy = Z - Z_center;

    if (sign * (fit - last_fit) < 0.0) {
      ds *= beta;
      continue;
    }

    R_center = R;
    Z_center = Z;
    last_fit = fit;
    ds *= alpha;

    if (dfit <= tol ||
        (std::abs(dx) <= tol && std::abs(dy) <= tol)) {
      return ErrorCode::Success;
    }
  }

  std::cerr << "[ERROR]: Solution did not converge quickly enough" << std::endl;
  fit = last_fit;
  return ErrorCode::Success;
}


