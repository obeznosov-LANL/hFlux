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

import jax.numpy as jnp


def _half_sums(coeffs):
    p = coeffs.shape[-1]
    if p <= 1:
        zeros = jnp.zeros(coeffs.shape[:-1], dtype=coeffs.dtype)
        return zeros, zeros
    k = jnp.arange(1, p, dtype=coeffs.dtype)
    nonconst = coeffs[..., 1:]
    return jnp.sum(nonconst * (0.5 ** k), axis=-1), jnp.sum(nonconst * ((-0.5) ** k), axis=-1)


def _eval_z(coeffs, zeta: float):
    p = coeffs.shape[-1]
    powers = zeta ** jnp.arange(p, dtype=coeffs.dtype)
    return jnp.sum(coeffs * powers, axis=-1)


def clean_divergence(hermite_data, hR: float, hZ: float, component0: int = 0):
    """Clean RB_Z using RB_R.

    Layout is `(iR, iZ, component, idR, idZ)`. `component0` identifies
    the start of a magnetic-field component block: `component0 + 0` is
    `R*B_R`, `component0 + 1` is `R*B_phi`, and `component0 + 2` is
    `R*B_Z`.
    """

    hd = jnp.asarray(hermite_data)
    RBR = hd[:, :, component0, :, :]
    RBZ_old = hd[:, :, component0 + 2, :, :]
    nR, nZ, Pr, Pz = RBZ_old.shape
    PrBR, PzBR = RBR.shape[2], RBR.shape[3]
    iZ0 = nZ // 2

    RBZ = jnp.zeros_like(RBZ_old)
    RBZ = RBZ.at[:, :, :, 0].set(RBZ_old[:, iZ0 : iZ0 + 1, :, 0])
    for idR in range(Pr):
        for k in range(1, Pz):
            if idR + 1 < PrBR and k - 1 < PzBR:
                val = -RBR[:, :, idR + 1, k - 1] * (hZ / hR) * (idR + 1) / k
                RBZ = RBZ.at[:, :, idR, k].set(val)

    sum_plus0, sum_minus0 = _half_sums(RBZ[:, iZ0, :, :])
    a0_center = RBZ_old[:, iZ0, :, 0]
    boundary_top = a0_center + sum_plus0
    boundary_bottom = a0_center + sum_minus0
    RBZ = RBZ.at[:, iZ0, :, 0].set(a0_center)

    boundary = boundary_top
    for iZ in range(iZ0 + 1, nZ):
        sum_plus, sum_minus = _half_sums(RBZ[:, iZ, :, :])
        a0 = boundary - sum_minus
        RBZ = RBZ.at[:, iZ, :, 0].set(a0)
        boundary = a0 + sum_plus

    boundary = boundary_bottom
    for iZ in range(iZ0 - 1, -1, -1):
        sum_plus, sum_minus = _half_sums(RBZ[:, iZ, :, :])
        a0 = boundary - sum_plus
        RBZ = RBZ.at[:, iZ, :, 0].set(a0)
        boundary = a0 + sum_minus

    return hd.at[:, :, component0 + 2, :, :].set(RBZ)


def compute_flux(hermite_data, hR: float, hZ: float, component0: int = 0):
    """Compute psi Hermite coefficients.

    Layout is `(iR, iZ, component, idR, idZ)`. `component0` identifies
    the start of a magnetic-field component block: `component0 + 0` is
    `R*B_R`, `component0 + 1` is `R*B_phi`, and `component0 + 2` is
    `R*B_Z`.
    """

    hd = jnp.asarray(hermite_data)
    RBR = hd[:, :, component0, :, :]
    RBZ = hd[:, :, component0 + 2, :, :]
    nR, nZ, Pr, Pz = RBZ.shape
    PpsiR = Pr
    PpsiZ = Pz
    iR0 = nR // 2
    iZ0 = nZ // 2

    psi = jnp.zeros((nR, nZ, PpsiR, PpsiZ), dtype=hd.dtype)

    for idR in range(PpsiR):
        for idZ in range(1, PpsiZ):
            if idR < RBR.shape[2] and idZ - 1 < RBR.shape[3]:
                psi = psi.at[:, :, idR, idZ].set(-hZ * RBR[:, :, idR, idZ - 1] / idZ)

    for idR in range(PpsiR):
        sum_plus, sum_minus = _half_sums(psi[:, iZ0, idR, :])
        a0 = -sum_minus
        psi = psi.at[:, iZ0, idR, 0].set(a0)
        boundary = a0 + sum_plus
        for iZ in range(iZ0 + 1, nZ):
            sum_plus, sum_minus = _half_sums(psi[:, iZ, idR, :])
            a0 = boundary - sum_minus
            psi = psi.at[:, iZ, idR, 0].set(a0)
            boundary = a0 + sum_plus

        boundary = jnp.zeros((nR,), dtype=hd.dtype)
        for iZ in range(iZ0 - 1, -1, -1):
            sum_plus, sum_minus = _half_sums(psi[:, iZ, idR, :])
            a0 = boundary - sum_plus
            psi = psi.at[:, iZ, idR, 0].set(a0)
            boundary = a0 + sum_minus

    for idR in range(1, PpsiR):
        if idR - 1 < Pr:
            z_anchor = _eval_z(RBZ[:, iZ0, idR - 1, :], -0.5)
            psi = psi.at[:, :, idR, 0].add((hR * z_anchor / idR)[:, None])

    coeffs_R = []
    for idR in range(1, PpsiR):
        if idR - 1 < Pr:
            coeffs_R.append(hR * _eval_z(RBZ[:, iZ0, idR - 1, :], -0.5) / idR)
        else:
            coeffs_R.append(jnp.zeros((nR,), dtype=hd.dtype))
    if coeffs_R:
        coeffs_R = jnp.stack(coeffs_R, axis=1)
        sum_plus = jnp.sum(coeffs_R * (0.5 ** jnp.arange(1, PpsiR, dtype=hd.dtype))[None, :], axis=1)
        sum_minus = jnp.sum(coeffs_R * ((-0.5) ** jnp.arange(1, PpsiR, dtype=hd.dtype))[None, :], axis=1)

        a0 = -sum_minus[iR0]
        psi = psi.at[iR0, :, 0, 0].add(a0)
        boundary = a0 + sum_plus[iR0]
        for iR in range(iR0 + 1, nR):
            a0 = boundary - sum_minus[iR]
            psi = psi.at[iR, :, 0, 0].add(a0)
            boundary = a0 + sum_plus[iR]

        boundary = jnp.asarray(0.0, dtype=hd.dtype)
        for iR in range(iR0 - 1, -1, -1):
            a0 = boundary - sum_plus[iR]
            psi = psi.at[iR, :, 0, 0].add(a0)
            boundary = a0 + sum_minus[iR]

    return psi
