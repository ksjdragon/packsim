import numpy as np, os, csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from squish import Simulation, DomainParams, ordered
from squish.common import OUTPUT_DIR
from script_tools import (
    RC_SETTINGS,
    get_args,
    get_data,
    get_simulation_data,
    get_ordered_data,
)

NAME = "Cumulative-CellVEE"
ALPHA = 1.0


def main():
    sims_path, regen = get_args(
        "Anti-cumulative distribution of VEE and percent of equilibria for fixed alpha",
        "folders that contains various N simulations to plot",
    )

    sim, frames = Simulation.load(sims_path)
    vees = np.linspace(-3, 3, 10000)
    e_hex = ordered.e_hex(sim.domain)
    energies = {"all": np.empty(0, dtype=float)}
    counts = {}
    for frame in frames:
        energy = frame["stats"]["site_energies"] - e_hex
        defects = np.count_nonzero(frame["stats"]["site_edge_count"] != 6)
        if defects not in energies:
            energies[defects] = np.empty(0, dtype=float)
        energies[defects] = np.append(energies[defects], energy)
        energies["all"] = np.append(energies["all"], energy)

    all_count = None
    for defect, energy in energies.items():
        count = np.empty(vees.shape, dtype=float)
        for i, vee in enumerate(vees):
            count[i] = np.count_nonzero(energy >= vee)
        count = 100 * count / len(energy)
        if defect == "all":
            all_count = count
        else:
            counts[defect] = count

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    for defect, count in sorted(counts.items()):
        if defect == min(counts.keys()):
            ax.plot(vees, count, label=fr"$\mathrm{{D}}={defect}$", color="C0")
        elif defect == max(counts.keys()):
            ax.plot(vees, count, label=fr"$\mathrm{{D}}={defect}$", color="C1")
        else:
            ax.plot(vees, count, linestyle="dotted", alpha=0.3)
    ax.plot(vees, all_count, label="All", color="black")

    ax.set_xlim(-2.5, 2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel(r"VEE")
    ax.set_ylabel("Percent of Voronoi Regions")

    ax.grid(zorder=0)
    ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    main()
