from __future__ import annotations
import matplotlib.pyplot as plt
import os, numpy as np
import cmath, math, pickle


def main():

    with open("site_verts.pkl", "rb") as f:
        sites, site_verts = pickle.load(f)

    for i in range(400):
        verts = [
            as_complex(site_verts[i][j] - sites[i]) for j in range(len(site_verts[i]))
        ]
        plot_2d(verts, f"squish_output/maxcenters_sim/{i:03}.png")
    return
    v = [
        0.266 + 0.87j,
        -0.626 + 0.747j,
        -0.976 - 0.046j,
        -0.283 - 0.873j,
        0.676 - 0.447j,
        0.875 + 0.414j,
    ]

    # v = [v[0], v[1], v[3]]

    line = np.linspace(-1, 1, 120)
    line2 = np.linspace(-1, 1, 120)
    X, Y = np.meshgrid(line, line2)
    Z = np.empty(X.shape)
    DZ = np.empty(X.shape, dtype="complex")
    HZ = np.empty((X.shape[0], X.shape[1], 2, 2))
    for i, x in enumerate(line):
        for j, y in enumerate(line2):
            rad, deriv, hess = average_radius(x + 1j * y, v, l)
            Z[j][i] = rad
            DZ[j][i] = deriv
            HZ[j][i] = hess

    max_indices = np.unravel_index(np.argmax(Z), Z.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.contour(
        X,
        Y,
        Z,
        np.linspace(4, 5.7, 15),
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    ax1.scatter(X[max_indices], Y[max_indices], Z[max_indices])

    cent = centroid(v, l)
    maxcent = maxcenter(v, l)
    ax1.scatter(cent.real, cent.imag, 3)
    ax1.scatter(maxcent.real, maxcent.imag, 3)

    print(maxcent)
    print(abs(maxcent - cent))

    ax1.view_init(elev=90, azim=270)
    plt.show()

    ax2 = fig.add_subplot(111, projection="3d")
    ax2.contour(
        X,
        Y,
        DZ.real,
        np.linspace(-3, 3, 9),
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    ax2.contour(
        X,
        Y,
        DZ.imag,
        np.linspace(-3, 3, 9),
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )

    ax2.view_init(elev=90, azim=270)
    ax2.scatter(X[max_indices], Y[max_indices], Z[max_indices])
    # ax2.plot_surface(X, Y, DZy, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    # for vert in v:
    #    ax.scatter(vert.real, vert.imag, 5)

    plt.savefig("TestPolygonAverageRadius.png")
    plt.show()


def plot_2d(v: List[complex], name: str):
    l = get_l(v)
    cent, maxcent = centroid(v, l), maxcenter(v, l)

    fig, ax = plt.subplots(1, figsize=(8, 8))

    for i in range(len(v)):
        va, vap = v[i], v[(i + 1) % len(v)]
        ax.plot([va.real, vap.real], [va.imag, vap.imag], color="black")

    ax.scatter(cent.real, cent.imag, label="Centroid")
    ax.scatter(maxcent.real, maxcent.imag, label="Maxcenter")
    ax.legend()
    ax.grid()

    plt.savefig(name)
    plt.close()


def get_l(v):
    l = []
    for i in range(len(v)):
        l.append(v[(i + 1) % len(v)] - v[i])
    return l


def generate_hexagon():
    angles = np.sort(np.random.random_sample((6,)))
    while np.any(np.diff(angles) >= 0.5):
        angles = np.sort(np.random.random_sample((6,)))
    angles *= 2 * math.pi

    mags = np.random.random_sample((3,))
    mags = np.array([mags[0], 1, mags[1], 1, mags[2], 1])

    v = []
    for mag, angle in zip(mags, angles):
        v.append(cmath.rect(1, angle))

    return v


def centroid(v, l):
    area, cent = 0, 0
    for i in range(len(v)):
        jdi = v[i] * 1j
        A = (jdi.conjugate() * l[i] + jdi * l[i].conjugate()) / 4
        area += A
        cent += (2 * v[i] + l[i]) * A

    return (1 / (3 * area)) * cent


def average_radius(x, v, l):
    radius, deriv, hess = [], [], []
    for i in range(len(v)):
        jdi = (v[i] - x) * 1j
        A = (jdi.conjugate() * l[i] + jdi * l[i].conjugate()) / 2
        k = -1j * l[i]

        da, dap = v[i] - x, v[i] - x + l[i]
        dau, dapu = da / abs(da), dap / abs(dap)
        kcu = k.conjugate() / abs(k)
        z, zp = kcu * dau, kcu * dapu

        int_rad = 2 * (cmath.atan(zp) - cmath.atan(z)) / (1j * abs(k))

        radius.append(A * int_rad)
        deriv.append(-k * int_rad)
        hess.append(np.dot(as_vector(k).T, as_vector(1j * (dapu - dau) / A)))

    if True in [x.real < 0 for x in radius]:
        return 0, 0, 0
    else:
        return sum(radius).real, sum(deriv), sum(hess)


def as_vector(c):
    return np.atleast_2d(np.array([c.real, c.imag]))


def as_complex(v):
    v = v.flatten()
    return v[0] + 1j * v[1]


def maxcenter(v, l, delta=1e-8):
    above_thres = True
    x = centroid(v, l)
    while above_thres:
        rad, deriv, hess = average_radius(x, v, l)
        above_thres = np.linalg.norm(deriv) > delta
        x -= as_complex(as_vector(deriv).dot(np.linalg.inv(hess)))
    return x


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*=false"
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
