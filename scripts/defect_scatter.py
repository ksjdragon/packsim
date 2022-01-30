import argparse, numpy as np, os, math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from squish import Simulation, ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS

NAME = "DefectScatter"


def main():
    parser = argparse.ArgumentParser(description="Graphs scatter of VEE and defects.")
    parser.add_argument(
        "sims_path", metavar="sim_dir", help="simulation to make scatter plot of"
    )

    args = parser.parse_args()

    sim, frames = Simulation.load(args.sims_path)
    defects, energy = [], []
    e_hex = ordered.e_hex(sim.domain)
    for frame in frames:
        defects.append(np.count_nonzero(frame["stats"]["site_edge_count"] != 6))
        energy.append(frame["energy"] / sim.domain.n - e_hex)

    defects = np.array(defects)
    energy = 100 * np.array(energy)

    plt.rcParams.update(RC_SETTINGS)
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    m, b = np.polyfit(defects, energy, 1)
    ax.scatter(defects, energy, color="C0", marker="*", s=100)
    ax.plot(np.arange(0, 64), np.arange(0, 64) * m + b, zorder=3, color="C1")

    ax.scatter(0, b, color="C2", s=500, marker="*", zorder=50)

    ax.annotate(
        r"$\zeta_0$",
        xy=(0, b),
        xytext=(10, -50),
        textcoords="offset points",
        # arrowprops={"arrowstyle": "->", "color": "black"},
    )

    ax.set_xlim(0, 64)
    ax.set_xticks(np.arange(0, 65, 8))

    ax.set_ylim(0, 3.6)
    ax.set_yticks(np.arange(0, 3.6, 0.5))

    ax.set_xlabel("Number of Defects")
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
