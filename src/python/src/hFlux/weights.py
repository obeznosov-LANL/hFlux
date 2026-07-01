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


def fdw(swidth: int, dtype=jnp.float64):
    """Finite-difference weights from hFlux/FiniteDifferenceWeights.hpp."""

    if swidth == 5:
        values = [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1.0 / 12],
            [-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12],
            [-0.5, 1.0, 0.0, -1.0, 0.5],
            [1.0, -4.0, 6.0, -4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    elif swidth == 7:
        values = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0 / 60, 3.0 / 20, -3.0 / 4, 0.0, 3.0 / 4, -3.0 / 20, 1.0 / 60],
            [1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20, 1.0 / 90],
            [1.0 / 8, -1.0, 13.0 / 8, 0.0, -13.0 / 8, 1.0, -1.0 / 8],
            [-1.0 / 6, 2.0, -13.0 / 2, 28.0 / 3, -13.0 / 2, 2.0, -1.0 / 6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    elif swidth == 9:
        values = [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 280, -4.0 / 105, 1.0 / 5, -4.0 / 5, 0.0, 4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280],
            [-1.0 / 560, 8.0 / 315, -1.0 / 5, 8.0 / 5, -205.0 / 72, 8.0 / 5, -1.0 / 5, 8.0 / 315, -1.0 / 560],
            [-7.0 / 240, 3.0 / 10, -169.0 / 120, 61.0 / 30, 0.0, -61.0 / 30, 169.0 / 120, -3.0 / 10, 7.0 / 240],
            [7.0 / 240, -2.0 / 5, 169.0 / 60, -122.0 / 15, 91.0 / 8, -122.0 / 15, 169.0 / 60, -2.0 / 5, 7.0 / 240],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    else:
        raise ValueError("unsupported swidth; expected 5, 7, or 9")

    return jnp.asarray(values, dtype=dtype)
