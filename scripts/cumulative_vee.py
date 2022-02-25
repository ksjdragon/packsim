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

NAME = "Cumulative-VEE"
ALPHA = 1.0


def main():
    sims_path, regen = get_args(
        "Anti-cumulative distribution of VEE and percent of equilibria for fixed alpha",
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

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    my_cool_data = [["VEE", 61, 67, 73, 81, 84, 100]]
    for vee in np.linspace(0, 0.06, 10000):
        my_cool_data.append([vee])

    for j, package in enumerate(packages):
        data, ordered_data, domain = package
        e_hex = ordered.e_hex(domain)

        alpha_index = np.where(data["all"]["alpha"] == ALPHA)[0][0]

        energies = data["all"]["Energy"][alpha_index] / domain.n - e_hex

        min_order = ordered_data["Energy"][alpha_index][0] / domain.n - e_hex

        vees = np.linspace(0, 0.06, 10000)
        index = np.argmin(np.abs(vees - min_order))

        counts = np.empty(vees.shape, dtype=float)
        for i, vee in enumerate(vees):
            counts[i] = np.count_nonzero(energies >= vee)
        counts = 100 * counts / len(energies)

        for i, count in enumerate(counts):
            my_cool_data[i + 1].append(count)

        ax.plot(
            100 * vees[: index + 1],
            counts[: index + 1],
            label=f"N={domain.n}",
            color=f"C{j}",
        )
        ax.plot(
            100 * vees[index:],
            counts[index:],
            label=f"_nolegend_",
            linestyle="dotted",
            color=f"C{j}",
        )

    with open("cumulative-vee.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(my_cool_data)

    ax.set_xlim(0, 6.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel(r"VEE $\left[\times 10^{2}\right]$")
    ax.set_ylabel("Percent of Equilibria")

    ax.grid(zorder=0)
    ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    main()
