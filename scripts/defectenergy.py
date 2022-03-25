from squish import Simulation
import matplotlib.pyplot as plt
import os, numpy as np


def main():
    sim, frames = Simulation.load(
        "squish_output/Radial[T]Search - N11-400 - 10.00x10.00 - 500/Radial[T]Search - N397 - 10.00x10.00"
    )

    defect, energy = [], []
    for frame_info in frames:
        defect.append(np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6))
        energy.append(sum(frame_info["stats"]["site_energies"][:400]))

    fig, ax = plt.subplots(1, figsize=(8, 8))
    plt.subplots_adjust(0.1, 0.12, 0.97, 0.9)

    fig.suptitle("Defects vs. Energy")
    ax.set_xlabel("Defects")
    ax.set_ylabel("Energy")
    ax.grid(zorder=0)
    ax.set_xticks(np.arange(0, 64, 2))
    ax.scatter(defect, energy, zorder=3, color="C0", marker="*")

    m, b = np.polyfit(defect, energy, 1)
    ax.plot(
        defect, np.array(defect) * m + b, zorder=3, color="C1", label=f"Slope: {m:.4f}"
    )
    ax.legend()
    fig.savefig("DefectEnergyN397-10.00.png")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
