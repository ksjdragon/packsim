from __future__ import annotations
from typing import List, Tuple, Dict
import argparse, math, numpy as np, os, pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from multiprocessing import Pool, cpu_count
from pathlib import Path

import squish.ordered as order
from squish import Simulation, DomainParams
from squish.common import Energy, OUTPUT_DIR


def axis_settings(ax, widths):
    ax.invert_xaxis()
    ax.grid(zorder=0)
    ax.set_xticks([round(w, 2) for w in widths[::-2]])
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    plt.subplots_adjust(0.07, 0.12, 0.97, 0.9)


def main():
    # Loading arguments.
    parser = argparse.ArgumentParser("Outputs ordered equilibria lowest eigenvalues.")
    parser.add_argument(
        "n_objects",
        metavar="N",
        type=int,
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

    widths = np.round(np.linspace(3.0, 10.0, 141), 2)

    values = []
    with open("coercivity_eigs.pkl", "rb") as f:
        store_data = pickle.load(f)

    for width in widths:
        eig_vals = []
        for config, eigs in store_data[width].items():
            zero_ind = np.where(np.isclose(eigs, 0, atol=1e-4))[0][0]
            if zero_ind == 0:
                eig_vals.append(eigs[2])
            else:
                eig_vals.append(eigs[0])

        values.append(min(eig_vals))

    # for i, width in enumerate(widths):
    # 	domain = DomainParams(args.n_objects, width, 10, 4.0)
    # 	eig_vals = []
    # 	store_data[width] = {}
    # 	configs = order.configurations(domain)
    # 	for j, config in enumerate(configs):
    # 		if config == (1,0):
    # 			continue
    # 		points = order.sites(domain, config)

    # 		hess = Energy("radial-t").mode(*domain, points).hessian(10e-5)
    # 		eigs = np.sort(np.linalg.eig(hess)[0])[::-1]
    # 		store_data[width][config] = eigs

    # 		zero_ind = np.where(np.isclose(eigs, 0))[0][0]
    # 		if zero_ind == 0:
    # 			eig_vals.append(eigs[2])
    # 		else:
    # 			eig_vals.append(eigs[0])

    # 		hashes = int(21*j/len(widths))
    # 		print(f'Generating at {width}, {i+1}/{len(widths)}... |{"#"*hashes}{" "*(20-hashes)}|' + \
    # 				f' {j+1}/{len(configs)} configs done.', flush=True, end='\r')

    # print(flush=True)

    # with open("coercivity_eigs.pkl", "wb") as f:
    # 	pickle.dump(store_data, f, pickle.HIGHEST_PROTOCOL)

    # return
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(0.07, 0.12, 0.97, 0.9)

    ax.plot(widths, values)

    ax.invert_xaxis()
    ax.grid(zorder=0)
    ax.set_xticks([round(w, 2) for w in widths[::-2]])
    ax.set_xticklabels(ax.get_xticks(), rotation=90)

    fig.suptitle("Coercivity")
    # ax.set_xlim([0, 5])
    ax.set_xlabel("Width")
    ax.set_ylabel("Eigenvalue")

    fig.savefig(OUTPUT_DIR / "Coercivity.png")
    print(f"Wrote to {OUTPUT_DIR / 'Coercivity.png'}")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
