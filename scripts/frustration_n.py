import numpy as np, os, math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from squish import Simulation, ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_args, get_data, format_data

NAME = "DefectEnergy"
NAME2 = "Frustration"


def proc(path):
    sim, frames = Simulation.load(path)
    domain = sim.domain
    e_hex = ordered.e_hex(domain)

    defects, energy = [], []
    for frame in frames:
        # if np.var(frame["stats"]["avg_radius"]) > 1e-8:
        defects.append(np.count_nonzero(frame["stats"]["site_edge_count"] != 6))
        energy.append(frame["energy"] / domain.n - e_hex)

    configs = ordered.configurations(domain)
    order_energies = []
    for config in configs:
        rbar = ordered.avg_radius(domain, config)

        order_energies.append(
            (
                2 * domain.w * domain.h
                + 2 * math.pi * domain.n * (domain.r ** 2 - 2 * domain.r * rbar)
            )
            / domain.n
            - e_hex
        )

    avg_defects = sum(defects) / len(defects)
    avg_vee = sum(energy) / len(energy)
    m, b = np.polyfit(defects, energy, 1)
    return [domain.n, avg_defects, avg_vee, m, b, min(order_energies)]


def main():
    sims_path, regen = get_args(
        "Generates graph for Average Defects and Energy per Defect for alpha=1",
        "folder that contains simulations at various N",
    )

    def f():
        data = {}

        files = list(sims_path.iterdir())
        with Pool(cpu_count()) as pool:
            for i, res in enumerate(pool.imap_unordered(proc, files)):
                data[res[0]] = res[1:]

                hashes = int(21 * i / len(files))
                print(
                    f'Processed N={res[0]} |{"#"*hashes}{" "*(20-hashes)}|'
                    + f" {i+1}/{len(files)} simulations processed.",
                    flush=True,
                    end="\r",
                )

            print(flush=True)

        return format_data(
            data,
            key_name="N",
            col_names=[
                "Average Defects",
                "Average VEE",
                "Energy Per Defect",
                "Ground VEE",
                "Minimum Ordered VEE",
            ],
        )

    plt.rcParams.update(RC_SETTINGS)
    data = get_data(sims_path / (NAME + ".pkl"), f, regen=regen)
    ns, avg_defects, avg_vees, epds, vee0s, min_order = (
        data["N"],
        data["Average Defects"],
        100 * data["Average VEE"],
        100 * data["Energy Per Defect"],
        100 * data["Ground VEE"],
        100 * data["Minimum Ordered VEE"],
    )

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    # ax2 = ax.twinx()
    # ax3 = ax.twinx()

    m0, b0 = np.polyfit(ns, avg_defects, 1)
    # m1, b1 = np.polyfit(ns, avg_defects * epds, 1)

    lns1 = ax.plot(
        ns, avg_defects, color="C0", alpha=0.5, label=r"$\langle \mathrm{D} \rangle$"
    )
    ax.plot(ns, m0 * ns + b0, color="C0", linestyle="dashed")

    lns2 = ax.plot(
        ns, 100 * epds, color="C1", alpha=0.5, label=r"$\zeta_1 \times 10^{4}$"
    )

    lns3 = ax.plot(
        ns,
        10 * avg_defects * epds,
        color="C2",
        alpha=0.5,
        label=r"$\mathcal{F} \times 10^{3}$",
    )
    # ax3.plot(ns, 100*(m1 * ns + b1) / 10, color="C2", linestyle="dashed")
    ax.set_xlim(93, 407)

    ax.set_ylim(3, 37)
    ax.set_yticks(np.arange(5, 40, 5))

    # ax2.set_ylim(-3 * 0.4, 18 + 3 * 0.4)
    # ax2.set_yticks(np.arange(0, 20, 3))

    # ax3.set_ylim(-3 * 0.5, 18 + 3 * 0.4)
    # ax3.set_yticks([])
    # ax3.spines["right"].set_visible(False)
    # ax3.spines.right.set_position(("axes", 1.11))

    # lns = lns1 + lns2 + lns3
    # labs = [l.get_label() for l in lns]
    ax.grid(zorder=0)
    # ax.legend(lns, labs, loc="lower right")
    ax.legend()
    ax.set_xlabel("N")
    # ax.set_ylabel(r"$\langle \mathrm{D} \rangle$")
    # ax2.set_ylabel("VEE")
    # ax2.set_ylabel(r"$\zeta \left[\times 10^{4} \right]$", color="C1")
    # ax3.set_ylabel(r"Defect Energy $\left[\times 10^{3} \right]$", color="C2")

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")

    # return
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.plot(ns, vee0s, label="$\zeta_0$", color="C5")
    ax.plot(ns, avg_defects * epds, label=r"$\mathcal{F}$", color="C2")
    ax.plot(ns, avg_vees, label=r"$\langle \mathrm{VEE} \rangle$", color="C4")
    ax.scatter(ns, min_order, label="Minimum Ordered", color="C7", alpha=0.8)

    print(np.linalg.norm(avg_vees - (vee0s + avg_defects * epds)))

    ax.grid(zorder=0)
    ax.legend()

    ax.set_xlim(95, 405)

    ax.set_ylim(-0.1, 9.2)
    ax.set_yticks(np.arange(0, 9.6, 1))

    ax.set_xlabel("N")
    ax.set_ylabel(r"VEE $\left[\times 10^{2}\right]$")

    fig.savefig(OUTPUT_DIR / (NAME2 + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME2 + '.png')}")


if __name__ == "__main__":
    os.environ["QT_log10GING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
