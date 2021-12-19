from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, math, numpy as np, os, pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
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

    return domain.w, min(energies), max(energies), min(isoparams), max(isoparams)


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


def probability_of_disorder(data, widths, domain):
    fig, ax = plt.subplots(figsize=(16, 8))
    all_disorder_count = []
    for width in widths:
        equal_shape = list([c[1] for c in data["all"][width]])
        all_disorder_count.append(
            100 * equal_shape.count(False) / len(data["all"][width])
        )

    return all_disorder_count


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

    return min_order - min_unorder


def sigmoid(x, x0, k):
    return 100 / (1 + np.exp(-k * (x - x0)))


def main():
    # Loading arguments.
    parser = argparse.ArgumentParser("Outputs width search data into diagrams")
    parser.add_argument(
        "sims_path",
        metavar="path/to/data",
        help="folder that contains simulation files of all searches for all N.",
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

    fig_folder = OUTPUT_DIR
    fig_folder.mkdir(exist_ok=True)

    store = Path(args.sims_path) / "EEvsPoD.pkl"

    if store.is_file():
        with open(store, "rb") as f:
            horiz, vert = pickle.load(f)
    else:
        horiz = []
        vert = []

        for file in Path(args.sims_path).iterdir():
            # Obtain data from simulation files and generate single shape data.
            data, widths, domain = get_equilibria_data(file)
            order_data = get_ordered_energies(domain, widths)

            vert.append(probability_of_disorder(data, widths, domain))
            horiz.append(excess_energy(data, widths, order_data, domain))

        horiz, vert = np.concatenate(horiz), np.concatenate(vert)
        with open(store, "wb") as f:
            pickle.dump((horiz, vert), f, pickle.HIGHEST_PROTOCOL)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(2):
        ax.scatter(
            horiz[i * 141 : (i + 1) * 141],
            vert[i * 141 : (i + 1) * 141],
            alpha=0.5,
            color=f"C{i}",
            s=5,
        )

        start, end = ax.get_xlim()

    # popt, pcov = curve_fit(sigmoid, horiz, vert)
    # x = np.linspace(start, end, 100)
    # y = sigmoid(x, *popt)
    # y = sigmoid(x, -1.35, 3)
    # ax.plot(x, y, color="C1")

    plt.subplots_adjust(0.1, 0.1, 0.97, 0.93)

    ax.set_xticks(np.linspace(start, end, 10))
    ax.set_yticks(np.arange(0, 105, 5))
    ax.grid()

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.title.set_text("Excess Energy Difference vs. PoD")
    ax.set_xlabel("Excess Energy Difference")
    ax.set_ylabel("Probability of Disorder")
    fig.savefig(OUTPUT_DIR / "Energy Diff and Probability")
    return

    # with open("testing.pkl", "rb") as f:
    # 	disorder_dict = pickle.load(f)
    # 	widths = np.linspace(3.0, 10.0, 141)
    # 	min_n, max_n = 60, 80

    disorder_dict = {}
    for file in Path(args.sims_path).iterdir():
        sim_data, widths, domain = get_equilibria_data(file)

        disorder_count = []
        for width in widths:
            equal_shape = list([c[1] for c in sim_data["all"][width]])
            disorder_count.append(
                100 * equal_shape.count(False) / len(sim_data["all"][width])
            )

        disorder_dict[domain.n] = disorder_count

    min_n, max_n = min(disorder_dict), max(disorder_dict)
    filepath = f"Disorder Heatmap N{min_n}-{max_n}"

    # with open("testing.pkl", "wb") as f:
    # 	pickle.dump(disorder_dict, f, pickle.HIGHEST_PROTOCOL)

    disorder_arr = np.zeros((max_n - min_n + 1, len(widths)))
    for key, value in disorder_dict.items():
        disorder_arr[key - min_n] = np.asarray(value)

    fig, ax = plt.subplots(figsize=(12, 8))

    extent = [min(widths), max(widths), min_n, max_n + 1]
    ax.imshow(
        disorder_arr,
        cmap="plasma",
        interpolation="nearest",
        aspect="auto",
        extent=extent,
    )

    ax.invert_xaxis()
    ax.set_xticks([round(w, 2) for w in widths[::-2]])
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_yticks(list(range(min_n, max_n + 1)))
    plt.subplots_adjust(0.07, 0.12, 0.97, 0.9)

    ax.title.set_text(filepath)
    ax.set_xlabel("Width")
    ax.set_ylabel("Number of Sites")
    fig.savefig(OUTPUT_DIR / filepath)

    print(f"Wrote to {OUTPUT_DIR / filepath}.")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
