import numpy as np, os
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

NAME = "VEEDiff-PoD"


def main():
    sims_path, regen = get_args(
        "Scatter plot of VEE difference and disordered equilibria.",
        "folders that contains various N simulations to plot",
    )

    packages = []
    for fol in sims_path.iterdir():
        if fol.is_file():
            continue

        data, n, r = get_data(
            fol / "package.pkl", get_simulation_data, args=(fol,), regen=regen
        )
        domain, alphas = DomainParams(n, 1, 1, r), data["all"]["alpha"]
        ordered_data = get_data(
            OUTPUT_DIR / "OrderedCache" / f"{n}.pkl",
            get_ordered_data,
            args=(domain, alphas),
            regen=regen,
        )

        packages.append([data, ordered_data, domain])

    packages.sort(key=lambda x: x[2].n)

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    for j, package in enumerate(packages):
        data, ordered_data, domain = package

        min_disorder, max_disorder = [], []
        for i, energies in enumerate(data["distinct"]["Energy"]):
            disorder_energies = []
            for j, energy in enumerate(energies):
                if not data["distinct"]["Ordered"][i][j]:
                    disorder_energies.append(energy)
            min_disorder.append(min(disorder_energies))
            max_disorder.append(max(disorder_energies))

        min_order = []
        for i, energies in enumerate(ordered_data["Energy"]):
            min_order.append(energies[0])

        e_hex = ordered.e_hex(domain)
        min_disorder = np.array(min_disorder) / domain.n - e_hex
        max_disorder = np.array(max_disorder) / domain.n - e_hex
        min_order = np.array(min_order) / domain.n - e_hex

        all_disorder_count = []
        for disorders in data["all"]["Ordered"]:
            all_disorder_count.append(
                100 * np.count_nonzero(disorders == False) / len(disorders)
            )

        ax.scatter(
            100 * (min_order - min_disorder), all_disorder_count, label=f"N={domain.n}"
        )

    ax.set_ylabel("POD")
    ax.set_xlabel(r"VEE Difference $\left[\times 10^{2}\right]$")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(48, 102)
    ax.set_yticks(np.arange(50, 101, 10))

    ax.grid(zorder=0)
    ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    main()
