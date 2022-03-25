import argparse, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt

from squish import ordered
from squish.common import OUTPUT_DIR, DomainParams, Energy
from squish.simulation import Simulation, Flow

from script_tools import RC_SETTINGS, get_data, format_data

NAME = "Perturbations"
NAME2 = "MinimumEscapeVEE"


def main():
    parser = argparse.ArgumentParser(
        description="Graphs perturbation graphs for a collection of simulations."
    )
    parser.add_argument(
        "sim_store_path", metavar="sim_dir", help="folder to save simulations to"
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
    out_fol = Path(args.sim_store_path)

    (OUTPUT_DIR / out_fol).mkdir(exist_ok=True)

    end_eq = Simulation.from_file(args.end_path)
    e_hex = ordered.e_hex(end_eq.domain)

    def f():
        all_data = []
        for j in range(50):
            pert_out_fol = out_fol / f"Vector{j:03}"
            (OUTPUT_DIR / pert_out_fol).mkdir(exist_ok=True)

            perturb = np.random.random_sample(end_eq.frames[0].site_arr.shape)
            perturb /= np.linalg.norm(perturb)

            data = {}
            k = -4
            same_eq = True
            while same_eq:
                print(f"Testing k={k} on vector {j:03}")
                k_out_fol = pert_out_fol / f"EQk{k}"
                delta = 10 ** k
                data[delta] = {"norm": [], "time": [], "vee": [], "k": k}
                if not pert_out_fol.is_dir():
                    this_pert = perturb * delta
                    sim = Flow(
                        end_eq.domain,
                        end_eq.energy,
                        end_eq.step_size,
                        end_eq.thres,
                        True,
                        name=k_out_fol,
                    )
                    sim.run(True, True, 50, end_eq.frames[0].site_arr + this_pert)

                sim, frames = Simulation.load(OUTPUT_DIR / k_out_fol)

                for i, frame in enumerate(frames):
                    adjusted = frame["arr"] + (
                        end_eq.frames[0].site_arr[0] - frame["arr"][0]
                    )

                    data[delta]["norm"].append(
                        np.linalg.norm(
                            ordered.toroidal_distance(
                                end_eq.domain, adjusted, end_eq.frames[0].site_arr
                            )
                        )
                    )
                    data[delta]["time"].append(sim.step_size * i)
                    data[delta]["vee"].append(frame["energy"] / sim.domain.n - e_hex)

                k += 1 if k < 0 else 0.25
                m, _ = np.polyfit(data[delta]["time"], data[delta]["norm"], 1)
                same_eq = m < 0

            all_data.append({"vec": perturb, "data": data})

        return all_data

    all_data = get_data(OUTPUT_DIR / out_fol / "PerturbData.pkl", f, regen=args.regen)

    end_vee = end_eq.frames[0].energy / end_eq.domain.n - e_hex
    print(end_vee)
    vees = []
    for dat in all_data:
        for k, v in sorted(dat["data"].items()):
            if v["norm"][-1] > 1e-3:
                vees.append(dat["data"][k]["vee"][0])
                break

    eigs = np.sort(np.linalg.eigvalsh(end_eq.frames[0].hessian))

    zero_ind = np.where(np.isclose(eigs, 0, atol=1e-8))[0]
    if len(zero_ind) == 0:
        coer = eigs[0]
    elif zero_ind[0] == 0:
        coer = eigs[2]
    else:
        coer = eigs[0]

    plt.rcParams.update(RC_SETTINGS)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.hist(vees, bins=np.linspace(0, 1, 30))
    ax.grid(zorder=0)

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, zorder=20)
    ax.text(
        0.60,
        0.96,
        f"Min Escape = {min(vees):.6f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

    ax.text(
        0.62,
        0.89,
        f"Coercivity = {coer:.6f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

    out = OUTPUT_DIR / f"{NAME2} - {args.sim_store_path}.png"
    fig.savefig(out)
    print(f"Wrote to {out}")

    return

    fig = plt.figure(figsize=(30, 8))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    for delta in sorted(data):
        ax.plot(
            np.array(data[delta]["time"]),
            np.array(data[delta]["norm"]),
            label=f"k = {data[delta]['k']}",
        )

    # ax.set_title(r"Relaxation of Perturbations")

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
