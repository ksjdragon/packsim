from __future__ import annotations
from typing import List
import argparse, pickle, numpy as np, math, os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from squish import DomainParams, Simulation, Energy, ordered
from squish.common import OUTPUT_DIR

RC_SETTINGS = {
    "axes.titlesize": 45,
    "axes.labelsize": 45,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.fontsize": 40,
    "lines.linewidth": 3,
    "font.family": "cm",
    "font.size": 40,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "figure.constrained_layout.use": True,
}


def get_args(
    script_desc: str, path_desc: str, cache_file: bool = True
) -> Tuple[Path, bool]:
    parser = argparse.ArgumentParser(description=script_desc)
    parser.add_argument("sims_path", metavar="sim_dir", help=path_desc)

    parser.add_argument(
        "--regenerate",
        dest="regen",
        action="store_true",
        help="regenerates the cache file for processed data",
    )

    args = parser.parse_args()

    return (Path(args.sims_path), args.regen)


def get_data(
    path: Path, func: Callable[Any, Any], args: Tuple[Any] = (), regen: bool = False
) -> Any:
    if regen:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    if path.is_file():
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        data = func(*args)
        with open(path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return data


def format_data(
    data: Dict[Any, Any], key_name: str, col_names=List[str]
) -> Dict[Any, Any]:
    data = sorted(data.items())
    new_data = {}
    new_data[key_name] = np.array([x[0] for x in data])
    for i, col_name in enumerate(col_names):
        col_value_type = type(data[0][1][i])
        if col_value_type is list or col_value_type is tuple:
            new_data[col_name] = [np.array(x[1][i]) for x in data]
        else:
            new_data[col_name] = np.array([x[1][i] for x in data])
    return new_data


def get_ordered_data(orig_domain: DomainParams, asps: np.ndarray) -> Dict:
    data = {}
    domains = []
    for alpha in asps:
        domains.append(
            [
                DomainParams(
                    orig_domain.n,
                    math.sqrt(orig_domain.n * alpha),
                    math.sqrt(orig_domain.n / alpha),
                    orig_domain.r,
                ),
                alpha,
            ]
        )

    with Pool(cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(ordered_data_proc, domains)):
            data[res[0]] = res[1:]

            hashes = int(21 * i / len(asps))
            print(
                f'Generating at width {res[0]:.02f}... |{"#"*hashes}{" "*(20-hashes)}|'
                + f" {i+1}/{len(asps)} completed.",
                flush=True,
                end="\r",
            )

        print(flush=True)

    return format_data(data, key_name="alpha", col_names=["Energy", "Coercivity"])


def ordered_data_proc(
    dom_tup: Tuple[DomainParams, float]
) -> Tuple[float, float, float, float]:
    domain, alpha = dom_tup
    energies, coercivities = [], []
    e_hex = ordered.e_hex(domain)

    configs = ordered.configurations(domain)
    for config in configs:
        # Causes errors, so ignore.
        if config[0] == 0 or config[1] == 0:
            continue
        # rbar = ordered.avg_radius(domain, config)
        # area = domain.w * domain.h / domain.n

        # energies.append(
        #    2 * domain.w * domain.h
        #    + 2 * math.pi * domain.n * (domain.r ** 2 - 2 * domain.r * rbar)
        # )

        # isoparams.append(math.pi * rbar ** 2 / area)

        sites = ordered.sites(domain, config)
        frame = Energy("radial-t").mode(*domain, sites)
        energies.append(frame.energy)

        eigs = np.sort(np.linalg.eigvalsh(frame.hessian))

        zero_ind = np.where(np.isclose(eigs, 0, atol=1e-8))[0]
        if len(zero_ind) == 0:
            coercivities.append(eigs[0])
        elif zero_ind[0] == 0:
            coercivities.append(eigs[2])
        else:
            coercivities.append(eigs[0])

    energies, coercivities = list(zip(*sorted(zip(energies, coercivities))))
    return (alpha, energies, coercivities)


def get_simulation_data(filepath: Path) -> Tuple[Dict, numpy.ndarray, DomainParams]:
    data = {"all": {}, "distinct": {}}
    files = list(Path(filepath).iterdir())

    with Pool(cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(simulation_data_proc, files)):
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

    data["all"] = format_data(
        data["all"], key_name="alpha", col_names=["Energy", "Ordered", "Defects"]
    )

    data["distinct"] = format_data(
        data["distinct"],
        key_name="alpha",
        col_names=["Energy", "Ordered", "Defects", "Hits"],
    )

    sim, frames = Simulation.load(files[0])
    return data, sim.domain.n, sim.domain.r


def simulation_data_proc(file: Path) -> Tuple[float, List[float], List[float]]:
    sim, frames = Simulation.load(file)

    alls = [[], [], []]
    for frame_info in frames:
        alls[0].append(frame_info["energy"])
        alls[1].append(np.var(frame_info["stats"]["avg_radius"]) <= 1e-8)
        alls[2].append(np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6))

    alls = list(zip(*sorted(zip(*alls))))

    sim, frames = Simulation.load(file)
    sim.frames = list(frames)
    counts = sim.get_distinct()

    distincts = [[], [], []]
    for j, frame_info in enumerate(sim.frames):
        distincts[0].append(frame_info["energy"])
        distincts[1].append(np.var(frame_info["stats"]["avg_radius"]) <= 1e-8)
        distincts[2].append(
            np.count_nonzero(frame_info["stats"]["site_edge_count"] != 6)
        )

    distincts.append(counts)
    distincts = list(zip(*sorted(zip(*distincts))))

    return sim.domain.w / sim.domain.h, alls, distincts
