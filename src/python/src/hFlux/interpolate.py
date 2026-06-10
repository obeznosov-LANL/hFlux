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

import jax
import jax.numpy as jnp

from .locator import StructuredLocator
from .weights import fdw


def compute_derivatives_stencil(view_data, D, ratioR: float, ratioZ: float, m: int):
    view_data = jnp.asarray(view_data)
    D = jnp.asarray(D, dtype=view_data.dtype)[: m + 1]

    base = D @ view_data @ D.T
    idx = jnp.arange(m + 1, dtype=view_data.dtype)
    fact = jnp.concatenate([jnp.ones((1,), dtype=view_data.dtype), jnp.cumprod(jnp.arange(1, m + 1, dtype=view_data.dtype))])
    sclx = (ratioR ** idx) / fact
    scly = (ratioZ ** idx) / fact
    return base * sclx[:, None] * scly[None, :]


def interpolate_in_place_1d(data, m: int):
    data = jnp.asarray(data)
    sz = m + 1
    n = 2 * sz
    if data.shape[0] != n:
        raise ValueError(f"expected length {n}, got {data.shape[0]}")

    NT = jnp.zeros((n, n), dtype=data.dtype)
    for i in range(sz):
        for idx in range(sz - i):
            NT = NT.at[i, idx].set(data[i])
            NT = NT.at[i, idx + sz].set(data[i + sz])

    for i in range(1, sz):
        for idx in range(sz - i, sz):
            NT = NT.at[i, idx].set(NT[i - 1, idx + 1] - NT[i - 1, idx])

    for i in range(sz, n):
        for idx in range(n - i):
            NT = NT.at[i, idx].set(NT[i - 1, idx + 1] - NT[i - 1, idx])

    out = NT[:, 0]
    for k in range(n - 2, -1, -1):
        sign = 1.0 if k < sz else -1.0
        for j in range(k, n - 1):
            out = out.at[j].add(0.5 * sign * out[j + 1])
    return out


def interpolate_2d(cell, m: int):
    """Interpolate one coefficient cell; input/output shape is (2*m+3, 2*m+3)."""

    cell = jnp.asarray(cell)
    n = 2 * (m + 1)
    if cell.shape != (n + 1, n + 1):
        raise ValueError(f"expected shape {(n + 1, n + 1)}, got {cell.shape}")

    block = cell[:n, :n]
    block = jax.vmap(lambda col: interpolate_in_place_1d(col, m), in_axes=1, out_axes=1)(block)
    block = jax.vmap(lambda row: interpolate_in_place_1d(row, m), in_axes=0, out_axes=0)(block)
    return cell.at[:n, :n].set(block)


def compute_derivatives_grid(data_component, fd_locator: StructuredLocator, hermite_locator: StructuredLocator, m: int, swidth: int):
    data_component = jnp.asarray(data_component)
    D = fdw(swidth, dtype=data_component.dtype)
    ratioR = hermite_locator.dR / fd_locator.dR
    ratioZ = hermite_locator.dZ / fd_locator.dZ
    stride = swidth - 1
    p = 2 * m + 3

    out = jnp.zeros((hermite_locator.nR, hermite_locator.nZ, p, p), dtype=data_component.dtype)
    for iR in range(hermite_locator.nR):
        for iZ in range(hermite_locator.nZ):
            for offR in range(2):
                for offZ in range(2):
                    ii = (iR + offR) * stride
                    jj = (iZ + offZ) * stride
                    idx0 = (m + 1) * offR
                    idz0 = (m + 1) * offZ
                    stencil = data_component[ii : ii + swidth, jj : jj + swidth]
                    derivs = compute_derivatives_stencil(stencil, D, ratioR, ratioZ, m)
                    out = out.at[iR, iZ, idx0 : idx0 + m + 1, idz0 : idz0 + m + 1].set(derivs)
    return out


def interpolate(data, fd_locator: StructuredLocator, hermite_locator: StructuredLocator, m: int = 2, swidth: int = 7):
    """Build Hermite coefficients from grid data.

    JAX layout is (iR, iZ, component, idR, idZ), unlike C++'s
    (idR, idZ, component, iR, iZ).
    """

    data = jnp.asarray(data)
    if data.ndim != 3:
        raise ValueError("data must have shape (nR_data, nZ_data, ncomp)")

    components = []
    for comp in range(data.shape[2]):
        coeffs = compute_derivatives_grid(data[:, :, comp], fd_locator, hermite_locator, m, swidth)
        coeffs = jax.vmap(jax.vmap(lambda cell: interpolate_2d(cell, m), in_axes=0), in_axes=0)(coeffs)
        components.append(coeffs)
    return jnp.stack(components, axis=2)
