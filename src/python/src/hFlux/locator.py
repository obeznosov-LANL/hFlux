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

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class StructuredLocator:
    """Uniform R-Z grid locator matching hFlux's C++ StructuredLocator."""

    R0: float
    Z0: float
    dR: float
    dZ: float
    nR: int
    nZ: int

    @property
    def R1(self) -> float:
        return self.R0 + self.nR * self.dR

    @property
    def Z1(self) -> float:
        return self.Z0 + self.nZ * self.dZ

    def check_bounds(self, R, Z):
        return (R >= self.R0) & (R <= self.R1) & (Z >= self.Z0) & (Z <= self.Z1)

    def locate(self, R, Z):
        sR = (R - self.R0) / self.dR
        sZ = (Z - self.Z0) / self.dZ
        iR = jnp.floor(sR).astype(jnp.int32)
        iZ = jnp.floor(sZ).astype(jnp.int32)
        xiR = sR - iR - 0.5
        xiZ = sZ - iZ - 0.5
        iR = jnp.clip(iR, 0, self.nR - 1)
        iZ = jnp.clip(iZ, 0, self.nZ - 1)
        return iR, iZ, xiR, xiZ


def make_hermite_locator(fd: StructuredLocator, swidth: int) -> StructuredLocator:
    if swidth < 3 or (swidth - 1) % 2 != 0:
        raise ValueError("swidth must be odd and at least 3")

    stride = swidth - 1
    margin = stride // 2
    return StructuredLocator(
        R0=fd.R0 + margin * fd.dR,
        Z0=fd.Z0 + margin * fd.dZ,
        dR=fd.dR * stride,
        dZ=fd.dZ * stride,
        nR=fd.nR // stride - 1,
        nZ=fd.nZ // stride - 1,
    )
