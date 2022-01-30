from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, math, numpy as np, os, pickle, itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from squish import Simulation, DomainParams
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_args, get_data, get_simulation_data


def main():
    sims_path, regen = get_args(
        "Density of states of various N across alpha",
        "folders that contains various N simulations to plot",
    )

    packages = []
    for fol in sims_path.iterdir():
        if fol.is_file():
            continue

        data, n, r = get_data(
            fol / "package.pkl", get_simulation_data, args=(fol,), regen=regen
        )
        domain = DomainParams(n, 1, 1, r)

        packages.append([data, domain])

    packages.sort(key=lambda x: x[1].n)

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    for j, package in enumerate(packages):
        data, domain = package

        distinct_disordered = []
        for ordered in data["distinct"]["Ordered"]:
            distinct_disordered.append(np.count_nonzero(ordered == False))

        ax.plot(data["all"]["alpha"], distinct_disordered, label=f"N={domain.n}")

    alphas = packages[0][0]["all"]["alpha"]
    ax.set_xticks([round(w, 2) for w in alphas[::10]])
    ax.set_xticklabels([f"{round(w, 3):.2f}" for w in alphas[::10]], rotation=90)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Number of Distinct Equilibria")

    ax.grid(zorder=0)
    ax.legend(loc="center right", fancybox=True, bbox_to_anchor=(1.34, 0.5))
    fig.savefig(OUTPUT_DIR / "DoS.png")

    print(f"Wrote to {OUTPUT_DIR / 'DoS.png'}")


if __name__ == "__main__":
    main()
