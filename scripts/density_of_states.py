from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, math, numpy as np, os, pickle, itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from squish import Simulation, DomainParams
from squish.common import OUTPUT_DIR


def main():
    # Loading arguments.
    parser = argparse.ArgumentParser("Outputs width search data into diagrams")
    parser.add_argument(
        "sims_path",
        metavar="path/to/data",
        help="Path to simulation folders with generated data.pkl from aspect_diagrams.py",
    )

    packages = []

    args = parser.parse_args()
    for fol in Path(args.sims_path).iterdir():
        if fol.is_file():
            continue

        store = fol / "data.pkl"
        if store.is_file():
            with open(store, "rb") as f:
                packages.append(pickle.load(f))
        else:
            print(
                f"{store} not found! Use aspect_diagrams.py to generate this file first."
            )

    if len(packages) == []:
        print("No data.pkl files found, terminating")
        return

    plt.rcParams.update(
        {
            "axes.titlesize": 45,
            "axes.labelsize": 45,
            "xtick.labelsize": 40,
            "ytick.labelsize": 40,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.fontsize": 40,
            "lines.linewidth": 3.5,
            "font.family": "cm",
            "font.size": 40,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "figure.constrained_layout.use": True,
        }
    )

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    packages.sort(key=lambda x: x[0].n)

    for j, package in enumerate(packages):
        domain, data, order_data, widths = package

        distinct_unordered = []
        for width in widths:
            equal_shape = list([c[1] for c in data["distinct"][width]])
            distinct_unordered.append(equal_shape.count(False))

        ax.plot(widths, distinct_unordered, label=f"N={domain.n}")

    widths = packages[0][3]
    ax.set_xticks([round(w, 2) for w in widths[::10]])
    ax.set_xticklabels([f"{round(w, 3):.2f}" for w in widths[::10]], rotation=90)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Number of Distinct Equilibria")

    ax.grid(zorder=0)
    ax.legend(loc="center right", fancybox=True, bbox_to_anchor=(1.34, 0.5))
    fig.savefig(OUTPUT_DIR / "DoS.png")

    print(f"Wrote to {OUTPUT_DIR / 'DoS.png'}")


if __name__ == "__main__":
    main()
