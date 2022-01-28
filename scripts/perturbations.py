import argparse, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt

from squish import DomainParams, Simulation
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_data, format_data

NAME = "Perturbations"


def toroidal_distance(domain: DomainParams, x: np.ndarray, y: np.ndarray) -> float:
    dim = np.array([domain.w, domain.h])
    absdist = np.abs(y - x)
    wrap = dim * (absdist >= np.array([domain.w, domain.h]) / 2)
    return np.linalg.norm(wrap - absdist)


def main():
    parser = argparse.ArgumentParser(
        description="Graphs perturbation graphs for a collection of simulations."
    )
    parser.add_argument(
        "sims_path",
        metavar="sim_dir",
        help="folder that contains of perturbations from an equilibrium",
    )
    parser.add_argument(
        "end_path",
        metavar="eq_path",
        help="simulation that contains the equilibrium to compare to.",
    )

    parser.add_argument(
        "--regenerate",
        dest="regen",
        action="store_true",
        help="regenerates the cache file for processed data",
    )

    args = parser.parse_args()
    sims_path = Path(args.sims_path)

    end_sim = Simulation.from_file(args.end_path)

    def f():
        data = {}
        for file in sims_path.iterdir():
            if "k" not in file.name or file.is_file():
                continue
            k = float(file.name.split("k")[-1])
            delta = 10 ** k

            sim, frames = Simulation.load(file)
            data[delta] = {"norm": [], "time": [], "k": k}

            for i, frame in enumerate(frames):
                adjusted = frame["arr"] + (
                    end_sim.frames[0].site_arr[0] - frame["arr"][0]
                )

                data[delta]["norm"].append(
                    toroidal_distance(
                        end_sim.domain, adjusted, end_sim.frames[0].site_arr
                    )
                )
                data[delta]["time"].append(sim.step_size * i)

        return data

    data = get_data(sims_path / "PerturbData.pkl", f, regen=args.regen)

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(30, 8))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    for delta in sorted(data):
        ax.plot(
            np.array(data[delta]["time"]),
            np.array(data[delta]["norm"]),
            label=f"k = {data[delta]['k']}",
        )

    ax.set_title(r"Relaxation of Perturbations")

    ax.set_xlim([0, 60])
    ax.set_yscale("log")

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"$\|\mathbf{x}-\mathbf{x_e}\|_2$")

    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1])
    ax.grid(zorder=0)

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
