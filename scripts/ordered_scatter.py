import argparse, numpy as np, os, math
import matplotlib.pyplot as plt

from squish import Simulation, ordered, DomainParams
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_args, get_data, get_ordered_data

NAME = "OrderedScatter"


def main():
    parser = argparse.ArgumentParser(
        description="Graphs scatter of ordered energy at fixed N."
    )
    parser.add_argument("n", metavar="N", type=int, help="N to make scatter plot of")
    parser.add_argument("r", metavar="R", type=float, help="natural radius of object")
    parser.add_argument(
        "--regenerate",
        dest="regen",
        action="store_true",
        help="regenerates the cache file for processed data",
    )
    args = parser.parse_args()

    ordered_data = get_data(
        OUTPUT_DIR / "OrderedCache" / f"{args.n}.pkl",
        get_ordered_data,
        args=(DomainParams(args.n, 1, 1, args.r), np.linspace(0.3, 1, 141)),
        regen=args.regen,
    )

    plt.rcParams.update(RC_SETTINGS)
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    e_hex = ordered.e_hex(DomainParams(args.n, 1, 1, args.r))
    for alpha, energies in zip(ordered_data["alpha"], ordered_data["Energy"]):
        ax.scatter(
            [alpha] * len(energies), 100 * (energies / args.n - e_hex), color="C0"
        )

    ax.set_xlim(0.3, 1)
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))

    ax.set_ylim(-0.1, 10)

    # ax.set_ylim(0, 3.6)
    # ax.set_yticks(np.arange(0, 3.6, 0.5))

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, zorder=20)
    ax.text(
        0.873,
        0.97,
        f"N={args.n}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel(r"VEE $\left[\times 10^{2}\right]$")
    ax.grid(zorder=0)

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    os.environ["QT_log10GING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
