import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot a Poincare section from a trace file "
        "(columns: trace turn R Z)."
    )
    parser.add_argument("poincare_file", help="Poincare text file")
    parser.add_argument(
        "-o", "--out", default=None, help="output PNG (default: <poincare_file>.png)"
    )
    args = parser.parse_args()

    out_path = args.out or (os.path.splitext(args.poincare_file)[0] + ".png")

    data = np.loadtxt(args.poincare_file, comments="#")
    trace = data[:, 0].astype(int)
    R = data[:, 2]
    Z = data[:, 3]

    fig, ax = plt.subplots(figsize=(7, 9))
    sc = ax.scatter(R, Z, c=trace, cmap="tab20", s=1, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("Poincare section")
    fig.colorbar(sc, ax=ax, label="trace id")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path} ({len(R)} points)")


if __name__ == "__main__":
    main()
