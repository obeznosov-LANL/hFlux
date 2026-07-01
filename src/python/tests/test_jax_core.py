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

import pathlib
import sys

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hFlux as hf


def analytic_field(R, Z, q0=2.1, q2=2.0, R_a=3.0):
    q = q0 + q2 * (R - R_a) ** 2 + q2 * Z**2
    return jnp.stack((-Z / q / R, R_a / R, (R - R_a) / q / R), axis=-1)


def test_fdw_7_matches_centered_stencil():
    D = hf.fdw(7)
    assert D.shape == (6, 7)
    assert D[0, 3] == pytest.approx(1.0)
    assert D[1, 0] == pytest.approx(-1.0 / 60.0)
    assert D[2, 3] == pytest.approx(-49.0 / 18.0)


def test_make_hermite_locator_matches_cpp_formula():
    fd = hf.StructuredLocator(R0=1.0, Z0=-2.0, dR=0.1, dZ=0.2, nR=30, nZ=42)
    hermite = hf.make_hermite_locator(fd, swidth=7)

    assert hermite.R0 == pytest.approx(1.3)
    assert hermite.Z0 == pytest.approx(-1.4)
    assert hermite.dR == pytest.approx(0.6)
    assert hermite.dZ == pytest.approx(1.2)
    assert hermite.nR == 4
    assert hermite.nZ == 6


def test_constant_field_interpolation_shape_and_eval():
    data = jnp.ones((31, 43, 3), dtype=jnp.float64)
    data = data.at[:, :, 1].set(2.0)
    data = data.at[:, :, 2].set(3.0)

    field = hf.FieldInterpolation.from_grid(data, R0=1.0, Z0=-2.0, dR=0.1, dZ=0.2, m=2, swidth=7)

    assert field.hermite_data.shape == (4, 6, 3, 7, 7)

    R = field.hermite_locator.R0 + 0.25 * field.hermite_locator.dR
    Z = field.hermite_locator.Z0 + 0.25 * field.hermite_locator.dZ
    B = field.eval_field(R, Z)

    assert B.shape == (3,)
    assert B[0] == pytest.approx(1.0)
    assert B[1] == pytest.approx(2.0)
    assert B[2] == pytest.approx(3.0)


def test_batched_eval_field():
    data = jnp.ones((31, 43, 3), dtype=jnp.float64)
    field = hf.FieldInterpolation.from_grid(data, R0=1.0, Z0=-2.0, dR=0.1, dZ=0.2, m=2, swidth=7)

    R = jnp.linspace(field.hermite_locator.R0 + 0.1, field.hermite_locator.R1 - 0.1, 5)
    Z = jnp.linspace(field.hermite_locator.Z0 + 0.1, field.hermite_locator.Z1 - 0.1, 5)
    B = field.vmap_eval_field(R, Z)

    assert B.shape == (5, 3)
    assert jnp.allclose(B, 1.0)


def test_magnetic_axis_and_flux_normalization():
    R0 = 1.525
    Z0 = -2.975
    dR0 = 0.0345
    dZ0 = 0.02975
    R1 = R0 + 99 * dR0
    Z1 = Z0 + 199 * dZ0
    nR = 31
    nZ = 43
    dR = (R1 - R0) / (nR - 1)
    dZ = (Z1 - Z0) / (nZ - 1)

    R = R0 + dR * jnp.arange(nR)
    Z = Z0 + dZ * jnp.arange(nZ)
    RR, ZZ = jnp.meshgrid(R, Z, indexing="ij")
    data = analytic_field(RR, ZZ) * RR[..., None]

    field = hf.FieldInterpolation.from_grid(data, R0=R0, Z0=Z0, dR=dR, dZ=dZ, m=2, swidth=7)
    field = field.compute_flux().clean_divergence()
    field, axis = field.normalize_flux(R_start=3.2, Z_start=-0.7, sign=-1)

    assert axis.converged
    assert axis.R == pytest.approx(3.0, abs=2.0e-3)
    assert axis.Z == pytest.approx(0.0, abs=2.0e-3)
    assert float(field.eval_psi(axis.R, axis.Z)) == pytest.approx(0.0, abs=1.0e-10)


def test_dopri5_integrates_exponential_growth():
    y = hf.solve_dopri5(lambda _t, state: state, jnp.array([1.0]), 0.0, 1.0, h=1.0e-3)

    assert y[0] == pytest.approx(float(jnp.e), rel=1.0e-8)


def test_poincare_trace_matches_analytic_circular_orbit():
    R0 = 1.525
    Z0 = -2.975
    dR0 = 0.0345
    dZ0 = 0.02975
    R1 = R0 + 99 * dR0
    Z1 = Z0 + 199 * dZ0
    nR = 31
    nZ = 43
    dR = (R1 - R0) / (nR - 1)
    dZ = (Z1 - Z0) / (nZ - 1)

    R = R0 + dR * jnp.arange(nR)
    Z = Z0 + dZ * jnp.arange(nZ)
    RR, ZZ = jnp.meshgrid(R, Z, indexing="ij")
    data = analytic_field(RR, ZZ) * RR[..., None]
    field = hf.FieldInterpolation.from_grid(data, R0=R0, Z0=Z0, dR=dR, dZ=dZ, m=2, swidth=7)

    seed = jnp.array([[3.2, 0.0]])
    poincare = field.compute_poincare(seed, n_turn=2, h=1.0e-3)

    assert poincare.shape == (1, 3, 2)
    assert poincare[0, -1, 0] == pytest.approx(float(seed[0, 0]), abs=1.0e-4)
    assert poincare[0, -1, 1] == pytest.approx(float(seed[0, 1]), abs=1.0e-4)
