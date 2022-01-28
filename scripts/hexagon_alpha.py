import argparse, numpy as np, os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from squish import ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_data, format_data

NAME = "RegHexTessRatios"


def get_ratios(n: int):
    return (n, ordered.hexagon_alpha(n))


def main():
    parser = argparse.ArgumentParser(
        description="Generates graph for regular hexagonal tessellation ratios"
    )
    parser.add_argument(
        "max_n", metavar="N", type=int, help="maximum N  of which to calculate"
    )

    parser.add_argument(
        "--regenerate",
        dest="regen",
        action="store_true",
        help="regenerates the cache file for processed data",
    )

    args = parser.parse_args()

    def f():
        data = {"alpha": [], "N": []}

        with Pool(cpu_count()) as pool:
            for i, res in enumerate(
                pool.imap_unordered(get_ratios, range(2, args.max_n + 1))
            ):
                for ratio in res[1]:
                    data["N"].append(res[0])
                    data["alpha"].append(ratio)

                hashes = int(21 * i / (args.max_n - 1))
                print(
                    f'Processed N={res[0]} |{"#"*hashes}{" "*(20-hashes)}|'
                    + f" {i+1}/{args.max_n-1}",
                    flush=True,
                    end="\r",
                )

            print(flush=True)
        return data

    data = get_data(OUTPUT_DIR / "OrderedCache" / (NAME + ".pkl"), f, regen=args.regen)

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    alphas, ns = [], []
    for alpha, n in zip(data["alpha"], data["N"]):
        if n <= args.max_n:
            alphas.append(alpha)
            ns.append(n)

    ax.scatter(alphas, ns, s=5)

    ax.set_xlim(-0.01, 1.1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0, args.max_n)

    ax.set_ylabel(r"$N$")
    ax.set_xlabel("Aspect Ratio")
    ax.grid(zorder=0)

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))

    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    os.environ["QT_log10GING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
