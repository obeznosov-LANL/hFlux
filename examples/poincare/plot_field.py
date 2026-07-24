import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot B_R, B_phi, B_Z on the R-Z plane from a fine-mesh "
        "field dump (columns: R Z phi B_R B_phi B_Z)."
    )
    parser.add_argument("field_file", help="fine-mesh field text file")
    parser.add_argument(
        "-o", "--out", default=None, help="output PNG (default: <field_file>.png)"
    )
    parser.add_argument(
        "--phi-index",
        type=int,
        default=0,
        help="toroidal plane index to plot (default: 0)",
    )
    args = parser.parse_args()

    out_path = args.out or (os.path.splitext(args.field_file)[0] + ".png")

    data = np.loadtxt(args.field_file, comments="#")
    R = data[:, 0]
    Z = data[:, 1]
    phi = data[:, 2]
    B = data[:, 3:6]

    # Select a single toroidal plane.
    phi_vals = np.unique(phi)
    if not 0 <= args.phi_index < len(phi_vals):
        raise SystemExit(
            f"phi-index {args.phi_index} outside [0, {len(phi_vals)})"
        )
    phi_sel = phi_vals[args.phi_index]
    mask = phi == phi_sel
    Rp = R[mask]
    Zp = Z[mask]
    Bp = B[mask]

    # Rebuild the structured (nR, nZ) grid on this plane.
    R_axis = np.unique(Rp)
    Z_axis = np.unique(Zp)
    nR = len(R_axis)
    nZ = len(Z_axis)
    order = np.lexsort((Zp, Rp))  # R outer, Z inner
    grid = Bp[order].reshape(nR, nZ, 3)
    RR = Rp[order].reshape(nR, nZ)
    ZZ = Zp[order].reshape(nR, nZ)

    titles = ["B_R", "B_phi", "B_Z"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), constrained_layout=True)
    for k, ax in enumerate(axes):
        pcm = ax.pcolormesh(RR, ZZ, grid[:, :, k], shading="auto", cmap="viridis")
        ax.contour(RR, ZZ, grid[:, :, k], levels=15, colors="k", linewidths=0.4,
                   alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax.set_title(f"{titles[k]} (phi = {phi_sel:.4g})")
        fig.colorbar(pcm, ax=ax, shrink=0.8)

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path} ({nR}x{nZ} grid, {len(phi_vals)} phi planes)")


if __name__ == "__main__":
    main()
