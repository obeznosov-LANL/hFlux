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

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .field import FieldInterpolation, MagneticAxis
from .flux import clean_divergence, compute_flux
from .interpolate import interpolate
from .locator import StructuredLocator, make_hermite_locator
from .taylor import eval_taylor
from .weights import fdw

__all__ = [
    "FieldInterpolation",
    "MagneticAxis",
    "StructuredLocator",
    "clean_divergence",
    "compute_flux",
    "eval_taylor",
    "fdw",
    "interpolate",
    "make_hermite_locator",
]
