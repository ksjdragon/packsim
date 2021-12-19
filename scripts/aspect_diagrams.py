from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, math, numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from multiprocessing import Pool, cpu_count
from pathlib import Path

import squish.ordered as order
from squish import Simulation, DomainParams
from squish.common import OUTPUT_DIR


def order_process(domain: DomainParams) -> Tuple[float, float, float]:
    energies, isoparams = [], []
    configs = order.configurations(domain)
    for config in configs:
        rbar = order.avg_radius(domain, config)
        area = domain.w * domain.h / domain.n

        energies.append(
            2 * domain.w * domain.h
            + 2 * math.pi * domain.n * (domain.r ** 2 - 2 * domain.r * rbar)
        )

        isoparams.append(math.pi * rbar ** 2 / area)

    return (domain.w, min(energies), max(energies), min(isoparams), max(isoparams))


def get_ordered_energies(orig_domain: DomainParams, widths: np.ndarray) -> Dict:
    data = {}
    domains = []
    for w in widths:
        aspect = w
        domains.append(
            DomainParams(
                orig_domain.n,
                math.sqrt(orig_domain.n * aspect),
                math.sqrt(orig_domain.n / aspect),
                orig_domain.r,
            )
        )

    # domains = [
    # DomainParams(orig_domain.n, w, orig_domain.h, orig_domain.r) for w in widths
    # ]

    with Pool(cpu_count()) as pool:
        energy_mins, energy_maxes, isoparam_mins, isoparam_maxes = {}, {}, {}, {}
        for i, res in enumerate(pool.imap_unordered(order_process, domains)):
            energy_mins[res[0]] = res[1]
            energy_maxes[res[0]] = res[2]
            isoparam_mins[res[0]] = res[3]
            isoparam_maxes[res[0]] = res[4]

            hashes = int(21 * i / len(widths))
            print(
                f'Generating at width {res[0]:.02f}... |{"#"*hashes}{" "*(20-hashes)}|'
                + f" {i+1}/{len(widths)} completed.",
                flush=True,
                end="\r",
            )

        print(flush=True)

        data["energy_min"] = list([x[1] for x in sorted(energy_mins.items())])
        data["energy_max"] = list([x[1] for x in sorted(energy_maxes.items())])
        data["isoparam_min"] = list([x[1] for x in sorted(isoparam_mins.items())])
        data["isoparam_max"] = list([x[1] for x in sorted(isoparam_maxes.items())])

    return data


def eq_file_process(file: Path) -> Tuple[float, List[float], List[float]]:
    sim, frames = Simulation.load(file)

    alls = []
    for frame_info in frames:
        alls.append(
            [
                frame_info["energy"],
                np.var(frame_info["stats"]["avg_radius"]) <= 1e-8,
                np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6),
                sum(frame_info["stats"]["site_energies"][: sim.domain.n]),
            ]
        )

    sim, frames = Simulation.load(file)
    sim.frames = list(frames)
    counts = sim.get_distinct()

    distincts = []
    for j, frame_info in enumerate(sim.frames):
        distincts.append(
            [
                frame_info["energy"],
                np.var(frame_info["stats"]["avg_radius"]) <= 1e-8,
                np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6),
                sum(frame_info["stats"]["site_energies"][: sim.domain.n]),
                counts[j],
            ]
        )

    return sim.domain.w / sim.domain.h, alls, distincts


def get_equilibria_data(filepath: Path) -> Tuple[Dict, numpy.ndarray, DomainParams]:
    data = {"all": {}, "distinct": {}}
    files = list(Path(filepath).iterdir())

    with Pool(cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(eq_file_process, files)):
            data["all"][res[0]] = res[1]
            data["distinct"][res[0]] = res[2]

            hashes = int(21 * i / len(files))
            print(
                f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|'
                + f" {i+1}/{len(files)} simulations loaded.",
                flush=True,
                end="\r",
            )
        print(flush=True)

    sim, frames = Simulation.load(files[0])
    widths = np.asarray(sorted(data["all"]))
    domain = DomainParams(sim.domain.n, widths[-1], sim.domain.h, sim.domain.r)
    return data, widths, domain


def axis_settings(ax, widths):
    ax.grid(zorder=0)
    ax.set_xticks([round(w, 2) for w in widths[::2]])
    ax.set_xticklabels([f"{round(w, 3):.2f}" for w in widths[::2]], rotation=90)
    plt.subplots_adjust(0.07, 0.12, 0.97, 0.9)


def probability_of_disorder(data, widths, domain):
    fig, ax = plt.subplots(figsize=(16, 8))
    all_disorder_count = []
    for width in widths:
        equal_shape = list([c[1] for c in data["all"][width]])
        all_disorder_count.append(
            100 * equal_shape.count(False) / len(data["all"][width])
        )

    ax.plot(widths, all_disorder_count)
    axis_settings(ax, widths)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.title.set_text(f"Probability of Disorder - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Disordered Equilibria")
    boa_y_min = round(min(all_disorder_count) / 20) * 20 - 5
    ax.set_yticks(np.arange(boa_y_min, 100.01, 2.5))

    return fig


def density_of_states(data, widths, domain):
    fig, ax = plt.subplots(figsize=(16, 8))
    distinct_ordered, distinct_unordered = [], []
    for width in widths:
        equal_shape = list([c[1] for c in data["distinct"][width]])
        distinct_ordered.append(equal_shape.count(True))
        distinct_unordered.append(equal_shape.count(False))

    ax2 = ax.twinx()
    ax.plot(widths, distinct_unordered, label="Unordered Equilibria", color="C0")
    ax2.plot(widths, distinct_ordered, label="Ordered Equilibria", color="C1")
    axis_settings(ax, widths)
    plt.subplots_adjust(0.07, 0.12, 0.92, 0.9)
    ax.title.set_text(f"Density of States - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Number of States (Disordered)", color="C0")
    ax2.set_ylabel("Number of States (Ordered)", color="C1")

    dos_y_max_unorder = 1.05 * max(distinct_unordered)
    dos_y_max_order = 1.05 * max(distinct_ordered)
    ax.set_yticks(np.linspace(0, dos_y_max_unorder, 20).astype(int))
    # ax.set_yticks(np.arange(0, dos_y_max_unorder, round(dos_y_max_unorder/200, 1)*10))
    ax2.set_yticks(np.arange(0, dos_y_max_order))

    return fig


def defect_density(data, widths, domain):
    fig, ax = plt.subplots(figsize=(16, 8))

    defects = []
    for width in widths:
        defects.append(
            sum([c[2] for c in data["all"][width] if not c[1]])
            / len(data["all"][width])
        )

    ax.plot(widths, defects)
    axis_settings(ax, widths)
    ax.title.set_text(f"Average Defects - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Defects")
    ax.set_yticks(np.arange(0, 1 + max(defects), 0.5))

    return fig


def circle_isoparam(data, widths, order_data, domain):
    fig, ax = plt.subplots(figsize=(16, 8))

    ax2 = ax.twinx()
    axis_settings(ax, widths)
    plt.subplots_adjust(0.07, 0.12, 0.92, 0.9)
    ax.title.set_text(f"Circular Isoparametric Ratio - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Maximum Ratio", color="C0")
    ax2.set_ylabel("Minimum Ratio", color="C1")

    ax.plot(widths, order_data["isoparam_max"], label="Maximum", color="C0")
    ax2.plot(widths, order_data["isoparam_min"], label="Minimum", color="C1")

    return fig


def reduced_energy(data, widths, order_data, domain):
    fig, ax = plt.subplots(figsize=(16, 8))

    ordered_energies, unordered_energies = [], []
    for width in widths:
        ordered_energies.append([c[0] for c in data["distinct"][width] if c[1]])
        unordered_energies.append([c[0] for c in data["distinct"][width] if not c[1]])

    for i in range(len(order_data["energy_min"])):
        ordered_energies[i].append(order_data["energy_min"][i])
        ordered_energies[i].append(order_data["energy_max"][i])

    min_order = np.asarray([min(width) for width in ordered_energies])
    max_order = np.asarray([max(width) for width in ordered_energies])
    min_unorder = np.asarray([min(width) for width in unordered_energies])
    max_unorder = np.asarray([max(width) for width in unordered_energies])

    offset = np.array(min_order)

    min_unorder_off = min_unorder - offset
    max_unorder_off = max_unorder - offset
    ax.plot(widths, min_order - offset, color="C1")
    # ax.plot(widths, max_order - offset, color='C1', linestyle='dotted')
    ax.plot(widths, min_unorder_off, color="C0")
    ax.plot(widths, max_unorder_off, color="C0", linestyle="dotted")
    axis_settings(ax, widths)

    ax.title.set_text(f"Reduced Energy vs. Width - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Reduced Energy")
    bif_y_max = np.max(np.abs(np.concatenate((min_unorder_off, max_unorder_off))))
    bif_top = np.arange(
        0, bif_y_max, round(bif_y_max / 20, -math.floor(math.log10(bif_y_max / 20)))
    )
    ax.set_yticks(np.concatenate((-bif_top[1:][::-1], bif_top)))

    return fig


def defect_energy(data, widths, order_data, domain):
    fig, ax = plt.subplots(figsize=(16, 8))

    ordered_energies, unordered_energies = [], []
    for width in widths:
        ordered_energies.append([c[0] for c in data["distinct"][width] if c[1]])
        unordered_energies.append([c[0] for c in data["distinct"][width] if not c[1]])

    for i in range(len(order_data["energy_min"])):
        ordered_energies[i].append(order_data["energy_min"][i])
        ordered_energies[i].append(order_data["energy_max"][i])

    min_order = np.asarray([min(width) for width in ordered_energies])
    max_order = np.asarray([max(width) for width in ordered_energies])
    min_unorder = np.asarray([min(width) for width in unordered_energies])
    max_unorder = np.asarray([max(width) for width in unordered_energies])

    offset = np.array(min_order)

    defect_a, defect_b = [], []
    for width in widths:
        num_defects = [c[2] for c in data["all"][width]]
        defect_energy = [c[3] for c in data["all"][width]]
        m, b = np.polyfit(num_defects, defect_energy, 1)

        defect_a.append(m)
        defect_b.append(b)

    ax2 = ax.twinx()
    ax.plot(widths, defect_a, label="Energy per Defect", color="C0")
    ax2.plot(widths, defect_b - offset, label="Relative Initial Energy", color="C1")
    axis_settings(ax, widths)
    plt.subplots_adjust(0.07, 0.12, 0.92, 0.9)
    ax.title.set_text(f"Defect Energy - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Energy per Defect", color="C0")
    ax2.set_ylabel("Relative Initial Energy", color="C1")

    return fig


def excess_energy(data, widths, order_data, domain):
    fig, ax = plt.subplots(figsize=(16, 8))

    ordered_energies, unordered_energies = [], []
    for width in widths:
        ordered_energies.append([c[0] for c in data["distinct"][width] if c[1]])
        unordered_energies.append([c[0] for c in data["distinct"][width] if not c[1]])

    for i in range(len(order_data["energy_min"])):
        ordered_energies[i].append(order_data["energy_min"][i])
        ordered_energies[i].append(order_data["energy_max"][i])

    min_order = np.asarray([min(width) for width in ordered_energies])
    max_order = np.asarray([max(width) for width in ordered_energies])
    min_unorder = np.asarray([min(width) for width in unordered_energies])
    max_unorder = np.asarray([max(width) for width in unordered_energies])

    # Energy of regular hexagon with area 1
    offset = (
        2
        - 2 * domain.r * (6 * 3 ** (-0.25) * math.sqrt(2) * math.atanh(0.5))
        + 2 * math.pi * domain.r ** 2
    )

    min_order_off = min_order / domain.n - offset
    min_unorder_off = min_unorder / domain.n - offset
    max_unorder_off = max_unorder / domain.n - offset

    ax.plot(widths, min_order_off, color="C1", label="Minimum Ordered")
    ax.plot(widths, min_unorder_off, color="C0", label="Minimum Disordered")
    ax.plot(
        widths,
        max_unorder_off,
        color="C0",
        linestyle="dotted",
        label="Maximum Disordered",
    )
    # ax.plot(
    #     [min(widths), max(widths)],
    #     [offset, offset],
    #     color="C1",
    #     linestyle="dotted",
    #     label="Regular Energy",
    # )

    axis_settings(ax, widths)
    ax.title.set_text(f"Energy at Aspect Ratios - N{domain.n}")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Excess Energy per Site")
    ax.legend()

    start, end = ax.get_ylim()
    ax.set_yticks(np.linspace(0, end, 20))
    ax.ticklabel_format(axis="y", style="sci")

    return fig


def main():
    # Loading arguments.
    parser = argparse.ArgumentParser("Outputs width search data into diagrams")
    parser.add_argument(
        "sims_path",
        metavar="path/to/data",
        help="folder that contains simulation files, or cached data file.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="suppress all normal output",
    )

    args = parser.parse_args()

    # Obtain data from simulation files and generate single shape data.
    data, widths, domain = get_equilibria_data(Path(args.sims_path))
    order_data = get_ordered_energies(domain, widths)

    fig_folder = OUTPUT_DIR / Path(f"AspectDiagrams - N{domain.n}")
    fig_folder.mkdir(exist_ok=True)

    # Generating diagrams.
    probability_of_disorder(data, widths, domain).savefig(
        fig_folder / "Probability of Disorder.png"
    )

    density_of_states(data, widths, domain).savefig(
        fig_folder / "Density Of States.png"
    )

    defect_density(data, widths, domain).savefig(fig_folder / "Defects.png")

    reduced_energy(data, widths, order_data, domain).savefig(
        fig_folder / "Reduced Energy.png"
    )

    defect_energy(data, widths, order_data, domain).savefig(
        fig_folder / "Defect Energy.png"
    )

    circle_isoparam(data, widths, order_data, domain).savefig(
        fig_folder / "Circular Isoparametric Ratio.png"
    )

    excess_energy(data, widths, order_data, domain).savefig(
        fig_folder / "Excess Energy.png"
    )

    print(f"Wrote to {fig_folder}.")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
