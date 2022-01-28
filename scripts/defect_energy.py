import numpy as np, os
import matplotlib.pyplot as plt

from squish import Simulation, ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_args, get_data, format_data

NAME = "DefectEnergy"


def main():
    sims_path, regen = get_args(
        "Generates graph for Average Defects and Energy per Defect.",
        "folder that contains simulations at various N",
    )

    def f():
        data = {}

        files = list(sims_path.iterdir())
        for i, file in enumerate(files):
            sim, frames = Simulation.load(file)
            domain = sim.domain
            e_hex = ordered.e_hex(domain)

            defects, energy = [], []
            for frame in frames:
                if np.var(frame["stats"]["avg_radius"]) > 1e-8:
                    defects.append(
                        np.count_nonzero(frame["stats"]["site_edge_count"] != 6)
                    )

                    energy.append(100 * frame["energy"] / domain.n)

            avg_defects = sum(defects) / (1 if len(defects) == 0 else len(defects))
            m, b = np.polyfit(defects, energy, 1)
            data[sim.domain.n] = [avg_defects, m]

            hashes = int(21 * i / len(files))
            print(
                f'Processed N={sim.domain.n:03} |{"#"*hashes}{" "*(20-hashes)}|'
                + f" {i+1}/{len(files)} simulations processed.",
                flush=True,
                end="\r",
            )

        print(flush=True)

        return format_data(
            data, key_name="N", col_names=["Average Defects", "Energy Per Defect"]
        )

    plt.rcParams.update(RC_SETTINGS)
    data = get_data(sims_path / (NAME + ".pkl"), f, regen=regen)
    ns, defects, epds = data["N"], data["Average Defects"], data["Energy Per Defect"]
    epds *= 100

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax2 = ax.twinx()
    ax3 = ax.twinx()

    m0, b0 = np.polyfit(ns, defects, 1)
    m1, b1 = np.polyfit(ns, defects * epds, 1)

    ax.plot(ns, defects, color="C0", alpha=0.5)
    ax.plot(ns, m0 * ns + b0, color="C0", linestyle="dashed")

    ax2.plot(ns, epds, color="C1", alpha=0.5)

    ax3.plot(ns, defects * epds / 10, color="C2", alpha=0.5)
    ax3.plot(ns, (m1 * ns + b1) / 10, color="C2", linestyle="dashed")

    ax.set_ylim(3, 37)
    ax.set_yticks(np.arange(5, 40, 5))

    ax2.set_ylim(-3 * 0.4, 18 + 3 * 0.4)
    ax2.set_yticks(np.arange(0, 20, 3))

    ax3.set_ylim(-3 * 0.5, 18 + 3 * 0.4)
    ax3.set_yticks([])
    ax3.spines["right"].set_visible(False)
    ax3.spines.right.set_position(("axes", 1.11))

    ax.grid(zorder=0)
    ax.set_xlabel("N")
    ax.set_ylabel("Average Defects", color="C0")
    ax2.set_ylabel(r"Energy per Defect $\left[\times 10^{4} \right]$", color="C1")
    ax3.set_ylabel(r"Defect Energy $\left[\times 10^{3} \right]$", color="C2")

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    os.environ["QT_log10GING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
