import numpy as np, os, csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit

from squish import Simulation, DomainParams, ordered
from squish.common import OUTPUT_DIR
from script_tools import (
    RC_SETTINGS,
    get_args,
    get_data,
    get_simulation_data,
    get_ordered_data,
)

NAME = "Cumulative-VEE-AcrossN"
NAME2 = "Cumulative-VEE-RelError"
NAME3 = "Cumulative-VEE-Alphas"


def main():
    sims_path, regen = get_args(
        "Anti-cumulative distribution of VEE and percent of equilibria for fixed alpha",
        "folders that contains various N simulations to plot",
    )

    vees = np.linspace(0, 0.06, 10000)

    def f(path):
        files = list(path.iterdir())
        files = [f for f in files if f.is_dir()]
        data = {}

        for i, fol in enumerate(files):
            sim, frames = Simulation.load(fol)
            e_hex = ordered.e_hex(sim.domain)
            energies = []
            for frame in frames:
                energies.append(frame["energy"] / sim.domain.n - e_hex)
            energies = np.array(energies)

            counts = np.empty(vees.shape, dtype=float)
            for j, vee in enumerate(vees):
                counts[j] = np.count_nonzero(energies >= vee)
            counts = 100 * counts / len(energies)

            data[sim.domain.n] = counts

            hashes = int(20 * i / len(files))
            print(
                f'Loading simulations... |{"#"*hashes}{" "*(20-hashes)}|'
                + f" {i+1}/{len(files)} simulations loaded.",
                flush=True,
                end="\r",
            )

        print(flush=True)
        return data

    all_data = {}
    alpha_files = list(sims_path.iterdir())
    alpha_files = [f for f in alpha_files if f.is_dir()]
    for fol in alpha_files:
        all_data[float(fol.name)] = get_data(
            fol / "Cumulative2.pkl", f, args=(fol,), regen=regen
        )

    data = all_data[1.0]
    points = []
    avgs = {}
    # print((data[106] + data[104]) / 2 - data[105])
    for n in range(105, 396):
        arr = np.zeros(data[100].shape)
        for i in range(1, 6):
            arr += data[n - i] + data[n + i]
        arr += data[n]
        arr /= 11
        avgs[n] = arr
        points.append(np.linalg.norm(data[n] - arr) / np.linalg.norm(arr))

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.plot(100 * vees, avgs[395], color="black", label=r"Average", zorder=50)

    for n, counts in sorted(data.items()):
        if n in range(390, 401):
            if n == 392:
                ax.plot(100 * vees, counts, label="N=392", color="C0")
            elif n == 397:
                ax.plot(100 * vees, counts, label="N=397", color="C1")
            elif n == 395:
                ax.plot(100 * vees, counts, color="black", linestyle="dashed")
            else:
                ax.plot(
                    100 * vees,
                    counts,
                    label="_nolegend_",
                    linestyle="dashed",
                    alpha=0.5,
                )

    ax.set_xlim(0, 6.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel(r"VEE $\left[\times 10^{2}\right]$")
    ax.set_ylabel("Percent of Equilibria")

    ax.grid(zorder=0)
    ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")

    fig = plt.figure(figsize=(30, 8))
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0])
    ns = np.array(range(105, 396))
    ax.scatter(ns, points)

    log_slope, _ = np.polyfit(np.log10(ns), np.log10(points), 1)
    print(log_slope)

    def g(x, k):
        return k * (x ** log_slope)

    # return a * np.e ** (-b * (x - c))

    params, covs = curve_fit(g, ns, points, p0=[5000])
    ax.plot(np.arange(50, 500), g(np.arange(50, 500), *params), color="C1")

    # with open("cumul.csv", "w") as csvfile:
    #    csvwriter = csv.writer(csvfile)
    #    csvwriter.writerows(zip(ns, points))
    print(params)
    ax.set_xlim(95, 405)
    ax.set_ylim(0, 0.24)
    ax.set_xlabel("N")

    ax.grid(zorder=0)
    # ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME2 + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME2 + '.png')}")

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    alpha_avgs = {}
    for alpha, data in all_data.items():
        if len(data.values()) == 0 or alpha == 1.0:
            continue
        alpha_avgs[alpha] = sum(data.values()) / 11

    for alpha, data in all_data.items():
        if alpha == 1.0:
            continue
        for n, counts in data.items():
            # ax.plot(100 * vees, counts, linestyle="dotted", alpha=0.4)
            continue

    pad = 100
    alpha_prob_dens = {}
    for alpha, avg in alpha_avgs.items():
        if alpha == 1.0:
            continue
        mov_avgs = np.array(
            [np.mean(avg[max(0, i - pad) : i + pad]) for i in range(len(avg))]
        )
        d_avgs = np.gradient(mov_avgs, vees[1] - vees[0])
        alpha_prob_dens[alpha] = d_avgs / (np.sum(d_avgs) * (vees[1] - vees[0]))

    a1_prob_dens = {}
    for n, avg in avgs.items():
        mov_avgs = np.array(
            [np.mean(avg[max(0, i - pad) : i + pad]) for i in range(len(avg))]
        )

        d_avgs = np.gradient(mov_avgs, vees[1] - vees[0])
        a1_prob_dens[n] = d_avgs / (np.sum(d_avgs) * (vees[1] - vees[0]))

    for alpha, prob_dens in sorted(alpha_prob_dens.items()):
        if alpha in [0.3, 0.5, 0.7, 0.9, 1.0]:
            continue
            ax.plot(100 * vees, prob_dens, label=fr"$\alpha={alpha:.1f}$")

    for n, prob_dens in sorted(a1_prob_dens.items()):
        if n in [255, 315, 335, 355, 375]:
            ax.plot(100 * vees, prob_dens, label=f"N={n}")
        if n == 395:
            ax.plot(100 * vees, prob_dens, label=f"N={n}", color="black")
            # ax.plot(100 * vees, prob_dens, label=r"$\alpha = 1.0$", zorder=50, color="black")
    # ax.plot(100 * vees, avgs[395], zorder=50, color="black")

    ax.set_xlim(0, 6.3)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel(r"VEE $\left[\times 10^{2}\right]$")
    # ax.set_ylabel("Percent of Equilibria")

    ax.grid(zorder=0)
    ax.legend()
    fig.savefig(OUTPUT_DIR / (NAME3 + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME3 + '.png')}")


if __name__ == "__main__":
    main()
