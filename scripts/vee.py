import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from squish import Simulation, DomainParams, Energy, ordered
from squish.common import OUTPUT_DIR
from script_tools import (
    RC_SETTINGS,
    get_args,
    get_data,
    get_simulation_data,
    get_ordered_data,
)

NAME = "VEE"


def main():
    sims_path, regen = get_args(
        "Various graphs for single N data.",
        "folder that contains simulation data at various aspect ratios for some N",
    )

    data, n, r = get_data(
        sims_path / "package.pkl", get_simulation_data, args=(sims_path,), regen=regen
    )
    domain, alphas = DomainParams(n, 1, 1, r), data["all"]["alpha"]

    if regen:
        os.remove(OUTPUT_DIR / "OrderedCache" / f"{n}".pkl)

    ordered_data = get_data(
        OUTPUT_DIR / "OrderedCache" / f"{n}.pkl",
        get_ordered_data,
        args=(domain, alphas),
        regen=regen,
    )

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

    hex_ratios = ordered.hexagon_alpha(domain.n)

    plt.rcParams.update(RC_SETTINGS)
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.plot(alphas, 100 * min_order, color="C1", label="Min Ordered", zorder=10)
    ax.plot(alphas, 100 * min_disorder, color="C0", label="Min Disordered")

    ax.plot(
        alphas,
        100 * max_disorder,
        color="C0",
        linestyle="dotted",
        label="Max Disordered",
    )

    ax.set_xlim(0.3, 1)

    # start, end = ax.get_ylim()
    # space = np.linspace(0, end, 20)

    space = np.arange(0, 6.6, 0.3)
    ax.set_ylim(-0.15, 6.6)
    ax.set_yticks(space[:-1])
    ax.ticklabel_format(axis="y", style="sci")

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel(r"VEE $\left[\times 10^{2}\right]$")

    ax.legend(loc=(0.23, 0.5))
    # ax.legend()
    ax.grid(zorder=0)

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, zorder=20)
    ax.text(
        0.873,
        0.97,
        f"N={domain.n}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

    out = OUTPUT_DIR / f"{NAME}-N{domain.n}.png"
    fig.savefig(out)
    print(f"Wrote to {out}.")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
