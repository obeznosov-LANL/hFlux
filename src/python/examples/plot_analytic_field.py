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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp


def analytic_field(R, Z, q0=2.1, q2=2.0, R_a=3.0):
    """AnalyticField::eval from test/AnalyticField.hpp."""

    q = q0 + q2 * (R - R_a) ** 2 + q2 * Z**2
    B_R = -Z / q / R
    B_phi = R_a / R
    B_Z = (R - R_a) / q / R
    return jnp.stack((B_R, B_phi, B_Z), axis=-1)


def analytic_psi(R, Z, q0=2.1, q2=2.0, R_a=3.0):
    q = q0 + q2 * (R - R_a) ** 2 + q2 * Z**2
    return 0.5 * jnp.log(q / q0) / q2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Interpolate and plot the analytic hFlux test field using the pure-JAX implementation."
    )
    parser.add_argument("--nR", type=int, default=100, help="number of input R grid points")
    parser.add_argument("--nZ", type=int, default=200, help="number of input Z grid points")
    parser.add_argument("--plot-nR", type=int, default=160, help="number of plotted R samples")
    parser.add_argument("--plot-nZ", type=int, default=320, help="number of plotted Z samples")
    parser.add_argument("--m", type=int, default=2, help="Hermite interpolation order parameter")
    parser.add_argument("--swidth", type=int, default=7, choices=(5, 7, 9), help="finite-difference stencil width")
    parser.add_argument("--poincare-rays", type=int, default=10, help="number of angular seed rays")
    parser.add_argument("--poincare-radial", type=int, default=10, help="number of radial seeds per ray")
    parser.add_argument("--poincare-turns", type=int, default=20, help="number of Poincare turns to trace")
    parser.add_argument("--poincare-h", type=float, default=1.0e-3, help="initial Dormand-Prince step for Poincare tracing")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("analytic_plots"), help="output directory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.poincare_rays <= 0:
        raise SystemExit("--poincare-rays must be positive")
    if args.poincare_radial <= 0:
        raise SystemExit("--poincare-radial must be positive")
    if args.poincare_turns < 0:
        raise SystemExit("--poincare-turns must be non-negative")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting: python -m pip install matplotlib") from exc

    try:
        import hFlux as hf
    except ModuleNotFoundError as exc:
        raise SystemExit("hFlux is not importable; run with PYTHONPATH=src/python/src or install the package") from exc

    R0 = 1.525
    Z0 = -2.975
    dR0 = 0.0345
    dZ0 = 0.02975
    R1 = R0 + 99 * dR0
    Z1 = Z0 + 199 * dZ0
    dR = (R1 - R0) / (args.nR - 1)
    dZ = (Z1 - Z0) / (args.nZ - 1)

    R_grid = R0 + dR * jnp.arange(args.nR)
    Z_grid = Z0 + dZ * jnp.arange(args.nZ)
    RR, ZZ = jnp.meshgrid(R_grid, Z_grid, indexing="ij")

    B = analytic_field(RR, ZZ)
    data = B * RR[..., None]

    print("Building Hermite interpolation...")
    field = hf.FieldInterpolation.from_grid(data, R0=R0, Z0=Z0, dR=dR, dZ=dZ, m=args.m, swidth=args.swidth)

    print("Computing flux...")
    field = field.compute_flux()
    print("Cleaning divergence...")
    field = field.clean_divergence()
    print("Finding magnetic axis and normalizing flux...")
    field, axis = field.normalize_flux(R_start=3.2, Z_start=-0.7, sign=-1)
    print(
        f"Magnetic axis: R={axis.R:.12g}, Z={axis.Z:.12g}, "
        f"psi={axis.psi:.12e}, iterations={axis.iterations}, converged={axis.converged}"
    )

    eps = 1.0e-8
    R_plot = jnp.linspace(field.hermite_locator.R0 + eps, field.hermite_locator.R1 - eps, args.plot_nR)
    Z_plot = jnp.linspace(field.hermite_locator.Z0 + eps, field.hermite_locator.Z1 - eps, args.plot_nZ)
    Rm, Zm = jnp.meshgrid(R_plot, Z_plot, indexing="ij")
    R_flat = Rm.reshape(-1)
    Z_flat = Zm.reshape(-1)

    print("Evaluating field and flux on plot grid...")
    eval_field = jax.jit(jax.vmap(field.eval_field))
    eval_psi = jax.jit(jax.vmap(field.eval_psi))
    RB_eval = eval_field(R_flat, Z_flat).reshape(args.plot_nR, args.plot_nZ, 3)
    B_eval = RB_eval / Rm[..., None]
    psi_eval = eval_psi(R_flat, Z_flat).reshape(args.plot_nR, args.plot_nZ)
    B_exact = analytic_field(Rm, Zm)
    psi_exact = analytic_psi(Rm, Zm)

    err_B = B_eval - B_exact
    psi_exact = psi_exact - analytic_psi(axis.R, axis.Z)
    err_psi = psi_eval - psi_exact
    l2_B = jnp.sqrt(jnp.mean(err_B**2, axis=(0, 1)))
    l2_psi = jnp.sqrt(jnp.mean(err_psi**2))
    print(f"RMS B error: BR={float(l2_B[0]):.6e}, Bphi={float(l2_B[1]):.6e}, BZ={float(l2_B[2]):.6e}")
    print(f"RMS normalized psi error: {float(l2_psi):.6e}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    R_np = jax.device_get(Rm)
    Z_np = jax.device_get(Zm)
    B_np = jax.device_get(B_eval)
    B_exact_np = jax.device_get(B_exact)
    psi_np = jax.device_get(psi_eval)
    psi_exact_np = jax.device_get(psi_exact)
    err_B_np = jax.device_get(err_B)
    err_psi_np = jax.device_get(err_psi)

    components = [("BR", 0), ("Bphi", 1), ("BZ", 2)]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for row, (name, comp) in enumerate(components):
        for ax, values, title in (
            (axes[row, 0], B_np[:, :, comp], f"Interpolated {name}"),
            (axes[row, 1], B_exact_np[:, :, comp], f"Exact {name}"),
            (axes[row, 2], err_B_np[:, :, comp], f"Error {name}"),
        ):
            im = ax.pcolormesh(R_np, Z_np, values, shading="auto")
            ax.set_title(title)
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            fig.colorbar(im, ax=ax)
    field_path = args.outdir / "analytic_field_components.png"
    fig.savefig(field_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, values, title in (
        (axes[0], psi_np, "Computed psi"),
        (axes[1], psi_exact_np, "Exact psi"),
        (axes[2], err_psi_np, "Psi error"),
    ):
        im = ax.pcolormesh(R_np, Z_np, values, shading="auto")
        ax.contour(R_np, Z_np, values, colors="k", linewidths=0.4, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax)
    psi_path = args.outdir / "analytic_flux.png"
    fig.savefig(psi_path, dpi=160)
    plt.close(fig)

    distance_to_edge = min(
        axis.R - field.hermite_locator.R0,
        field.hermite_locator.R1 - axis.R,
        axis.Z - field.hermite_locator.Z0,
        field.hermite_locator.Z1 - axis.Z,
    )
    rmax = (2.0 / 3.0) * distance_to_edge
    if rmax <= 0.01:
        raise SystemExit(f"not enough room for requested seed ray: rmax={rmax:.6e}")

    theta = jnp.arange(args.poincare_rays) * 2.0 * jnp.pi / args.poincare_rays
    radius = jnp.linspace(0.01, rmax, args.poincare_radial)
    dR_seed = radius[:, None] * jnp.cos(theta)[None, :]
    dZ_seed = radius[:, None] * jnp.sin(theta)[None, :]
    seeds_grid = jnp.stack((axis.R + dR_seed, axis.Z + dZ_seed), axis=-1)
    seeds = seeds_grid.reshape(-1, 2)

    print(
        f"Tracing Poincare data for {seeds.shape[0]} seeds "
        f"({args.poincare_rays} rays x {args.poincare_radial} radii), "
        f"{args.poincare_turns + 1} points including seed..."
    )
    compute_poincare = jax.jit(lambda x: field.compute_poincare(x, n_turn=args.poincare_turns, h=args.poincare_h))
    poincare = compute_poincare(seeds)
    expected_shape = (args.poincare_rays * args.poincare_radial, args.poincare_turns + 1, 2)
    if poincare.shape != expected_shape:
        raise RuntimeError(f"expected Poincare data shape {expected_shape}, got {poincare.shape}")

    poincare_np = jax.device_get(poincare)
    fig, ax = plt.subplots(figsize=(7, 9), constrained_layout=True)
    levels = 40
    contour = ax.contourf(R_np, Z_np, psi_np, levels=levels, cmap="viridis")
    ax.contour(R_np, Z_np, psi_np, levels=levels, colors="k", linewidths=0.25, alpha=0.35)
    fig.colorbar(contour, ax=ax, label="psi")
    colors = plt.cm.twilight(jax.device_get(jnp.linspace(0.0, 1.0, args.poincare_rays, endpoint=False)))
    for j in range(args.poincare_rays):
        ray_traces = poincare_np[j :: args.poincare_rays]
        for trace in ray_traces:
            ax.plot(trace[:, 0], trace[:, 1], ".", color=colors[j], ms=2.0)
    ax.plot(axis.R, axis.Z, "x", color="white", ms=7, mew=1.5, label="magnetic axis")
    ax.set_title(f"Poincare points on normalized psi ({args.poincare_turns + 1} points/seed)")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    poincare_path = args.outdir / "analytic_poincare_on_psi.png"
    fig.savefig(poincare_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {field_path}")
    print(f"Wrote {psi_path}")
    print(f"Wrote {poincare_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
