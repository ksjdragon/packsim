import argparse, numpy as np, os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from squish import Simulation, ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_data, format_data

NAME = "OrderedRBar"


def main():
    parser = argparse.ArgumentParser(
        description="Ordered rbar vector for specified frames of a simulation"
    )
    parser.add_argument(
        "sims_path",
        metavar="sim_dir",
        help="folder that contains of perturbations from an equilibrium",
    )
    parser.add_argument(
        "frames", metavar="frames", type=int, help="frame numbers to select", nargs="+"
    )

    args = parser.parse_args()

    rbars = [None] * len(args.frames)
    sim, frames = Simulation.load(args.sims_path)
    for i, frame in enumerate(frames):
        if i not in args.frames:
            continue

        rbars[args.frames.index(i)] = sorted(frame["stats"]["avg_radius"])

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(18, 14.4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    alphabet = "abcdefhijklmnopqrstuvwxyz"
    for i, rbar in enumerate(rbars):
        ax.plot(list(range(len(rbar))), rbar, label=alphabet[i])

    ax.set_xlim(-0.1, len(rbars[0]))
    ax.set_xticks(np.arange(0, 401, 40))
    # ax.set_ylim(0, args.max_n)

    ax.plot(
        list(range(len(rbars[0]))),
        [ordered.R_HEX] * (len(rbars[0])),
        linestyle="dotted",
        color="black",
    )

    ax.set_ylim(0.51, 0.615)
    ax.set_yticks(np.arange(0.52, 0.62, 0.03))
    ax.set_ylabel(r"$\bar{r}$")
    ax.grid(zorder=0)
    ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))

    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    os.environ["QT_log10GING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
