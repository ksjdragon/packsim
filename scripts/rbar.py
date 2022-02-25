import argparse, numpy as np, os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from squish import Simulation, ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_args, get_data, format_data

NAME = "RBar"


def main():
    sims_path, regen = get_args(
        "Variance of average radius for as N gets larger",
        "folders that contains various N simulations to plot",
    )

    def f():
        files = list(sims_path.iterdir())
        files = [f for f in files if f.is_dir()]
        data = {}

        for i, fol in enumerate(files):
            sim, frames = Simulation.load(fol)

            variances = []
            for frame in frames:
                rbars = np.sort(frame["stats"]["avg_radius"])
                variances.append(np.var(rbars))
            avg_variance = sum(variances) / len(variances)

            data[sim.domain.n] = [avg_variance]

            hashes = int(20 * i / len(files))
            print(
                f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|'
                + f" {i+1}/{len(files)} simulations loaded.",
                flush=True,
                end="\r",
            )

        print(flush=True)
        return format_data(data, key_name="N", col_names=["Average Variance"])

    data = get_data(sims_path / (NAME + ".pkl"), f, regen=regen)

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(18, 14.4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    primes = [
        i
        for i in range(min(data["N"]), max(data["N"]))
        if len(ordered.divisors(i)) == 2
    ]

    prime_points = []
    for i, n in enumerate(data["N"]):
        if n in primes:
            prime_points.append(data["Average Variance"][i])

    ax.plot(data["N"], 1e5 * data["Average Variance"])
    ax.scatter(primes, 1e5 * np.array(prime_points), marker="o", s=60)
    ax.set_xlim(95, 405)
    ax.set_ylim(0, 11)
    # ax.set_ylim(0, args.max_n)

    ax.set_xlabel("N")
    ax.set_ylabel(r"$\sigma \left[\times 10^5 \right]$")
    ax.grid(zorder=0)
    # ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))

    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    os.environ["QT_log10GING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
