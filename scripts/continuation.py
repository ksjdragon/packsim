import argparse, numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from multiprocessing import Pool, cpu_count
from pathlib import Path

from squish import Simulation, DomainParams, Energy, ordered, Diagram
from squish.common import OUTPUT_DIR
from script_tools import (
    RC_SETTINGS,
    get_args,
    get_data,
    get_simulation_data,
    get_ordered_data,
)

NAME = "Continuation"


def proc(combo):
    dia, min_order, min_disorder, hex_ratios, alphas, sim_alphas, sim_energies, offset, length, num_frames = (
        combo
    )
    sim = dia.sim
    out_folder = sim.path / "continuation"

    plt.rcParams.update(RC_SETTINGS)
    plt.rcParams.update({"figure.constrained_layout.use": False})
    fig = plt.figure(figsize=(30, 15))

    e_hex = ordered.e_hex(sim.domains[0])

    for i in range(length):
        gs = fig.add_gridspec(1, 2)
        gs.update(left=0, right=0.98, top=0.93, bottom=0.12, wspace=0.08)
        ax = fig.add_subplot(gs[1])

        curr_alpha = sim.domains[i].w / sim.domains[i].h
        curr_vee = 100 * (sim.energies[i] / sim.domains[i].n - e_hex)

        ax.plot(
            alphas,
            100 * min_order,
            color="C1",
            label="Min Ordered",
            zorder=10,
            linestyle="dotted",
        )
        ax.plot(
            alphas,
            100 * min_disorder,
            color="C0",
            label="Min Disordered",
            zorder=11,
            linestyle="dotted",
        )
        ax.plot(
            sim_alphas[:175],
            100 * sim_energies[:175],
            color="C2",
            label="_nolegend_",
            linestyle="dashed",
        )
        ax.plot(
            sim_alphas[174:], 100 * sim_energies[174:], color="C2", label="Continuation"
        )

        ax.scatter(
            hex_ratios, [0] * len(hex_ratios), color="C2", s=120, marker="H", zorder=50
        )

        ax.scatter(
            curr_alpha,
            curr_vee,
            s=250,
            facecolors="none",
            edgecolors="C6",
            linewidth=4,
            zorder=100,
        )

        ax.set_xlim(0.3, 1)
        ax.set_xticks([round(w, 2) for w in alphas[::10]])
        ax.set_xticklabels([f"{round(w, 3):.2f}" for w in alphas[::10]], rotation=90)

        # start, end = ax.get_ylim()
        # space = np.linspace(0, end, 20)

        space = np.arange(0, 10, 0.5)
        ax.set_ylim(-0.15, 9.5)
        ax.set_yticks(space[::-1])
        ax.ticklabel_format(axis="y", style="sci")

        ax.set_xlabel("Aspect Ratio")
        ax.set_ylabel(r"VEE $\left[\times 10^{2}\right]$")

        ax.legend(loc="upper center")
        # ax.legend()
        ax.grid(zorder=0)

        props = dict(boxstyle="round", facecolor="white", alpha=0.8, zorder=20)
        ax.text(
            0.873,
            0.97,
            f"N={sim.domains[i].n}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )

        ax1 = fig.add_subplot(gs[0])
        dia.voronoi_plot(i, ax1)
        fig.savefig(out_folder / f"img{i+offset:05}.png")
        fig.clear()

        j = len(list((sim.path / "continuation").iterdir()))
        hashes = int(21 * j / num_frames)
        print(
            f'Generating frames... |{"#"*hashes}{" "*(20-hashes)}|'
            + f" {j}/{num_frames} frames rendered.",
            flush=True,
            end="\r",
        )

    print(flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Makes video of analytic continuation."
    )
    parser.add_argument(
        "sims_path",
        metavar="sim_dir",
        help="folder that contains simulation data at various aspect ratios for some N",
    )
    parser.add_argument(
        "shrink_sim",
        metavar="continuation_sim",
        help="simulation that contains the continuation data",
    )
    args = parser.parse_args()
    sims_path = Path(args.sims_path)

    data, n, r = get_data(
        sims_path / "package.pkl", get_simulation_data, args=(sims_path,)
    )
    domain, alphas = DomainParams(n, 1, 1, r), data["all"]["alpha"]
    ordered_data = get_data(
        OUTPUT_DIR / "OrderedCache" / f"{n}.pkl",
        get_ordered_data,
        args=(domain, alphas),
    )

    min_disorder, max_disorder = [], []
    for i, energies in enumerate(data["distinct"]["Energy"]):
        disorder_energies = []
        for j, energy in enumerate(energies):
            if not data["distinct"]["Ordered"][i][j]:
                disorder_energies.append(energy)
        min_disorder.append(min(disorder_energies))
        max_disorder.append(max(disorder_energies))

    min_order, min_order_coer = [], []
    for i, energies in enumerate(ordered_data["Energy"]):
        min_order.append(energies[0])
        min_order_coer.append(ordered_data["Coercivity"][i][0])
    e_hex = ordered.e_hex(domain)
    min_disorder = np.array(min_disorder) / domain.n - e_hex
    max_disorder = np.array(max_disorder) / domain.n - e_hex
    min_order = np.array(min_order) / domain.n - e_hex

    hex_ratios = ordered.hexagon_alpha(domain.n)

    sim = Simulation.from_file(Path(args.shrink_sim))
    sim_alphas = np.array([x.w / x.h for x in sim.frames])
    sim_energies = np.array([x.energy for x in sim.frames]) / sim.domain.n - e_hex
    dia = Diagram(sim, ["voronoi"])

    out_folder = sim.path / "continuation"
    out_folder.mkdir(exist_ok=True)

    combo_list = []
    for i in range(cpu_count()):
        start, end = (
            int(i * len(sim) / cpu_count()),
            int((i + 1) * len(sim) / cpu_count()),
        )
        new_dia = Diagram(None, ["voronoi"])
        new_dia.sim = dia.sim.slice(list(range(start, end)))
        combo_list.append(
            (
                new_dia,
                min_order,
                min_disorder,
                hex_ratios,
                alphas,
                sim_alphas,
                sim_energies,
                start,
                len(sim.frames[start:end]),
                len(sim),
            )
        )

    print("Starting figure generation...", flush=True)
    with Pool(cpu_count()) as pool:
        for _ in pool.imap_unordered(proc, combo_list):
            pass

    video_path = sim.path / (NAME + ".mp4")

    fps = 30
    print("Assembling MP4...", flush=True)
    os.system(
        f"ffmpeg -hide_banner -loglevel error -r {fps} -i"
        + f' "{sim.path}/continuation/img%05d.png"'
        + f" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -vf"
        + f' "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 "{video_path}"'
    )

    print(f"Wrote to {video_path}.")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
