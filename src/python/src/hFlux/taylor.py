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


def eval_taylor(coeffs, x, y):
    """Evaluate sum_ij coeffs[..., i, j] * x**i * y**j."""

    coeffs = jnp.asarray(coeffs)
    px = coeffs.shape[-2]
    py = coeffs.shape[-1]
    xp = x ** jnp.arange(px, dtype=coeffs.dtype)
    yp = y ** jnp.arange(py, dtype=coeffs.dtype)
    return jnp.einsum("...ij,i,j->...", coeffs, xp, yp)
