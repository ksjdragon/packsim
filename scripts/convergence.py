from __future__ import annotations
from typing import List
import argparse, pickle, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt

from squish import Simulation
from squish.common import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        "Graphs convergence graphs for a collection of simulations."
    )
    parser.add_argument(
        "sims_path",
        metavar="path/to/data",
        help="folder that contains simulation files at various step sizes.",
    )

    args = parser.parse_args()
    data = {}

    for file in Path(args.sims_path).iterdir():
        sim, frames = Simulation.load(file)

        step = sim.step_size
        data[step] = {"times": [], "values": [], "diffs": []}
        for i, frame_info in enumerate(frames):
            data[step]["times"].append(step * i)
            data[step]["values"].append(np.linalg.norm(frame_info["arr"]))
            data[step]["diffs"].append(
                np.linalg.norm(all_info[-1]["arr"] - frame_info["arr"])
            )

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(0.07, 0.12, 0.97, 0.9)

    for step, d in data.items():
        ax[0].plot(d["times"], d["values"], label=step)
        ax[1].plot(d["times"], d["diffs"], label=step)

    fig.suptitle("Equilibrium Convergence")
    ax[0].grid(zorder=0)
    ax[0].legend()
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("L2 Norm of Sites")

    ax[1].grid(zorder=0)
    ax[1].legend()
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("L2 Norm of Difference")

    fig.savefig(OUTPUT_DIR / "Equilibrium Convergence.png")
    print(f"Wrote to {OUTPUT_DIR / 'Equilibrium Convergence.png'}")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
