import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from squish import ordered, DomainParams
from script_tools import RC_SETTINGS
from pathlib import Path

N = 64
C = (0, 0)

out_fol = Path(f"N{N}C{C}")
out_fol.mkdir(exist_ok=True)


def render(i, dom, vor):
    plt.rcParams.update(RC_SETTINGS)
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=False)
    ax.set_xlim(0, dom.w)
    ax.set_ylim(0, dom.h)
    fig.savefig(out_fol / f"{i:03}.png")


def get_full_points(dom, points):
    SYMM = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    return np.concatenate([points + dom.dim * s for s in SYMM])


def main():
    d = DomainParams(N, 8, 8, 4)
    # init = get_full_points(d, ordered.sites(d, C))
    init = get_full_points(d, np.random.random_sample((64, 2)) * d.dim)
    for i in range(10):
        vor = Voronoi(init if i == 0 else vor.vertices)
        render(i, d, vor)


if __name__ == "__main__":
    main()
