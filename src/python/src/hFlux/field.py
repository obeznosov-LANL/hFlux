#========================================================================================
# (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .flux import clean_divergence, compute_flux
from .interpolate import interpolate
from .locator import StructuredLocator, make_hermite_locator
from .taylor import eval_taylor


@dataclass(frozen=True)
class MagneticAxis:
    R: float
    Z: float
    psi: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class FieldInterpolation:
    """Pure-JAX hFlux field interpolation state.

    Array layouts:
    data: (nR_data, nZ_data, ncomp)
    hermite_data: (iR, iZ, component, idR, idZ)
    psi_data: (iR, iZ, idR, idZ)
    """

    fd_locator: StructuredLocator
    hermite_locator: StructuredLocator
    data: jnp.ndarray
    hermite_data: jnp.ndarray
    psi_data: jnp.ndarray | None
    m: int = 2
    swidth: int = 7

    @classmethod
    def from_grid(cls, data, R0: float, Z0: float, dR: float, dZ: float, m: int = 2, swidth: int = 7):
        data = jnp.asarray(data)
        if data.ndim != 3:
            raise ValueError("data must have shape (nR_data, nZ_data, ncomp)")
        fd_locator = StructuredLocator(R0, Z0, dR, dZ, data.shape[0] - 1, data.shape[1] - 1)
        hermite_locator = make_hermite_locator(fd_locator, swidth)
        hermite_data = interpolate(data, fd_locator, hermite_locator, m=m, swidth=swidth)
        return cls(fd_locator, hermite_locator, data, hermite_data, None, m, swidth)

    def clean_divergence(self, component0: int = 0):
        """Return a copy with divergence-cleaned magnetic-field coefficients.

        `component0` is the first component of a consecutive magnetic-field
        block: `component0 + 0` is `R*B_R`, `component0 + 1` is `R*B_phi`,
        and `component0 + 2` is `R*B_Z`.
        """

        hd = clean_divergence(self.hermite_data, self.hermite_locator.dR, self.hermite_locator.dZ, component0)
        return replace(self, hermite_data=hd)

    def compute_flux(self, component0: int = 0):
        """Return a copy with poloidal-flux Hermite coefficients.

        `component0` is the first component of a consecutive magnetic-field
        block: `component0 + 0` is `R*B_R`, `component0 + 1` is `R*B_phi`,
        and `component0 + 2` is `R*B_Z`.
        """

        psi = compute_flux(self.hermite_data, self.hermite_locator.dR, self.hermite_locator.dZ, component0)
        return replace(self, psi_data=psi)

    def eval_field(self, R, Z):
        iR, iZ, xiR, xiZ = self.hermite_locator.locate(R, Z)
        coeffs = self.hermite_data[iR, iZ, :, :, :]
        return eval_taylor(coeffs, xiR, xiZ)

    def eval_psi(self, R, Z):
        if self.psi_data is None:
            raise ValueError("psi_data is not available; call compute_flux() first")
        iR, iZ, xiR, xiZ = self.hermite_locator.locate(R, Z)
        coeffs = self.psi_data[iR, iZ, :, :]
        return eval_taylor(coeffs, xiR, xiZ)

    def find_magnetic_axis(
        self,
        R_start: float,
        Z_start: float,
        sign: int = -1,
        tol: float = 1.0e-9,
        max_iter: int = 100_000,
        alpha: float = 1.1,
        beta: float = 0.5,
        ds0: float = 0.5,
    ) -> MagneticAxis:
        """Find a psi extremum using the C++ hFlux gradient-descent logic.

        sign=-1 searches toward lower psi values; sign=1 searches toward higher
        psi values. The interpolated field stores R*B, matching the C++ tests.
        """

        if self.psi_data is None:
            raise ValueError("psi_data is not available; call compute_flux() first")
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or 1")
        if not bool(self.hermite_locator.check_bounds(R_start, Z_start)):
            raise ValueError("starting point is outside the Hermite interpolation domain")

        R = float(R_start)
        Z = float(Z_start)
        ds = float(ds0)
        last_fit = float(self.eval_psi(R, Z))

        for iteration in range(max_iter):
            if not bool(self.hermite_locator.check_bounds(R, Z)):
                raise ValueError("magnetic-axis search left the Hermite interpolation domain")

            B = self.eval_field(R, Z)
            gradx = float(B[2]) * R
            grady = -float(B[0]) * R
            grad = (gradx * gradx + grady * grady) ** 0.5
            if grad == 0.0:
                return MagneticAxis(R=R, Z=Z, psi=last_fit, iterations=iteration, converged=True)

            coeff = ds / grad
            R_next = R + sign * coeff * gradx
            Z_next = Z + sign * coeff * grady

            if not bool(self.hermite_locator.check_bounds(R_next, Z_next)):
                ds *= beta
                continue

            fit = float(self.eval_psi(R_next, Z_next))
            dfit = abs(fit - last_fit)
            dR = R_next - R
            dZ = Z_next - Z

            if sign * (fit - last_fit) < 0.0:
                ds *= beta
                continue

            R = R_next
            Z = Z_next
            last_fit = fit
            ds *= alpha

            if dfit <= tol or (abs(dR) <= tol and abs(dZ) <= tol):
                return MagneticAxis(R=R, Z=Z, psi=last_fit, iterations=iteration + 1, converged=True)

        return MagneticAxis(R=R, Z=Z, psi=last_fit, iterations=max_iter, converged=False)

    def normalize_flux(self, R_start: float, Z_start: float, sign: int = -1):
        """Shift psi so the magnetic-axis value is zero.

        This mirrors the C++ test normalization: subtract Psi_min from the
        constant coefficient of every psi cell.
        """

        if self.psi_data is None:
            raise ValueError("psi_data is not available; call compute_flux() first")
        axis = self.find_magnetic_axis(R_start, Z_start, sign=sign)
        psi = self.psi_data.at[:, :, 0, 0].add(-axis.psi)
        return replace(self, psi_data=psi), axis

    def vmap_eval_field(self, R, Z):
        return jax.vmap(self.eval_field)(R, Z)

    def vmap_eval_psi(self, R, Z):
        return jax.vmap(self.eval_psi)(R, Z)
