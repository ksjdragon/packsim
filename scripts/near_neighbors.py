import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from squish import Simulation, DomainParams, ordered
from squish.common import OUTPUT_DIR
from script_tools import RC_SETTINGS, get_args, get_data

NAME = "NearNeighbors"
second_neigh = True


def main():
    sims_path, regen = get_args(
        "Probability distribution of near neighbrs at distances",
        "simulation folder of equilibriums to plot",
    )

    dists = np.linspace(0, 5, 25000)

    def f():
        all_data = {}
        for n in [400]:
            if second_neigh:
                sim = Simulation.from_file(sims_path / f"Radial[T]Search - N{n} - 2500")
                print("Loaded simulation.")
            else:
                sim, frames = Simulation.load(
                    sims_path / f"Radial[T]Search - N{n} - 2500"
                )

            counts = np.empty(dists.shape, dtype=float)
            total_dists = np.array([], dtype=float)
            if second_neigh:
                for i, frame in enumerate(sim.frames):
                    if np.count_nonzero(frame.stats["site_edge_count"] != 6) != 6:
                        continue
                    total_dists = np.append(total_dists, frame.stats["site_distances"])
                    snn = frame.second_near_neighbor()

                    snn_dists = np.empty((sum([len(x) for x in snn])), dtype=float)
                    ind = 0
                    for j, site_snn in enumerate(snn):
                        site_snn_dists = ordered.toroidal_distance(
                            sim.domain,
                            frame.site_arr[site_snn],
                            np.ones((len(site_snn), 2)) * frame.site_arr[j],
                        )
                        snn_dists[ind : ind + len(site_snn_dists)] = site_snn_dists
                        ind += len(site_snn_dists)

                    total_dists = np.append(total_dists, snn_dists)

                    hashes = int(21 * i / len(sim))
                    print(
                        f'Processing N={n} at frame {i}... |{"#"*hashes}{" "*(20-hashes)}|'
                        + f" {i+1}/{len(sim)} completed.",
                        flush=True,
                        end="\r",
                    )
                print(flush=True)
            else:
                for frame in frames:
                    if (
                        np.count_nonzero(frame["stats"]["site_edge_count"] != 6) / n
                        < 0.08
                    ):
                        continue
                    total_dists = np.append(
                        total_dists, frame["stats"]["site_distances"]
                    )

            for i, dist in enumerate(dists):
                counts[i] = np.count_nonzero(total_dists <= dist)
            counts = 100 * counts / len(total_dists)

            pad = 20
            counts = np.array(
                [np.mean(counts[max(0, i - pad) : i + pad]) for i in range(len(counts))]
            )

            grad = np.gradient(counts, dists[1] - dists[0])
            all_data[n] = grad / (np.sum(grad) * (dists[1] - dists[0]))

        return all_data

    all_data = get_data(sims_path / "NearNeighborsD6.pkl", f, regen=regen)
    data_2 = get_data(sims_path / "NearNeighborsD60.pkl", f, regen=regen)

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    for n, counts in sorted(all_data.items()):
        ax.plot(dists, counts, label=f"D=6")

    for n, counts in sorted(data_2.items()):
        ax.plot(dists, counts, label=f"D=60")

        ax.set_xlim(0, 3)

    ax.set_xlabel(r"Distance")
    # ax.set_ylabel("Percent of Distances")

    ax.grid(zorder=0)
    ax.legend()

    fig.savefig(OUTPUT_DIR / (NAME + ".png"))
    print(f"Wrote to {OUTPUT_DIR / (NAME + '.png')}")


if __name__ == "__main__":
    main()
