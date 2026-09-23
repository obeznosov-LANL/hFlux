import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Poincare seed file: n_angles rays x n_radial "
        "points around the magnetic axis. Format: first line n_turn, then "
        "'R Z' per seed."
    )
    parser.add_argument("out", help="output seed file")
    parser.add_argument("--n-turn", type=int, default=1000, help="toroidal turns per seed")
    parser.add_argument("--n-angles", type=int, default=5, help="number of poloidal rays")
    parser.add_argument("--n-radial", type=int, default=100, help="radial points per ray")
    parser.add_argument("--r-max", type=float, default=0.2, help="maximum minor radius")
    parser.add_argument("--r-center", type=float, default=3.0, help="axis R")
    parser.add_argument("--z-center", type=float, default=0.0, help="axis Z")
    args = parser.parse_args()

    theta = np.arange(args.n_angles) * 2.0 * np.pi / args.n_angles
    # Skip r=0 (degenerate, all rays coincide at the axis).
    radius = np.linspace(args.r_max / args.n_radial, args.r_max, args.n_radial)

    with open(args.out, "w") as f:
        f.write(f"{args.n_turn}\n")
        for th in theta:
            for r in radius:
                R = args.r_center + r * np.cos(th)
                Z = args.z_center + r * np.sin(th)
                f.write(f"{R:.17g} {Z:.17g}\n")

    print(
        f"wrote {args.out} "
        f"({args.n_angles * args.n_radial} seeds, r_max={args.r_max})"
    )


if __name__ == "__main__":
    main()
