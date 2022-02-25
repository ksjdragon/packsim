import numpy as np
import matplotlib.pyplot as plt


def torus_mesh(n, r, R):
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def torus_line(n, r, R, u, v):
    phi = np.linspace(0, 2 * np.pi * u, n)
    theta = (v / u) * phi
    theta, phi = theta % (2 * np.pi), phi % (2 * np.pi)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    line = np.column_stack([x, y, z])
    x, y, z = R * np.cos(phi), R * np.sin(phi), 0 * phi
    norm = line - np.column_stack([x, y, z])

    return line, norm


def torus_points(r, R, u, v, k):
    phi = np.linspace(0, 2 * np.pi * u, k + 1)
    theta = (v / u) * phi
    theta, phi = theta % (2 * np.pi), phi % (2 * np.pi)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return x, y, z


def get_obscure_segs(arr):
    bounds = [0, 0]
    seg = 1
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            bounds[seg] = i
            bounds.append(i)
            seg += 1
    return bounds + [len(arr) - 1]


def main():
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")

    ax.set_box_aspect(aspect=(1, 1, 1))

    alpha = np.sqrt(3) * 2 / 5
    size = 5

    azim, elev = 0, 40
    R, r = size, size * (1 - alpha) / alpha
    u, v = 5, 2
    k = 20

    print(alpha, r / (r + R))

    ax.set_zlim(-(r + R), r + R)
    ax.view_init(elev, azim)

    ax.plot_surface(
        *torus_mesh(100, r, R), rcount=200, ccount=200, color="white", alpha=0.4
    )

    # Calculate when surface is facing or away and apply dotted vs solid lines.
    az, el = azim * np.pi / 180 - np.pi, elev * np.pi / 180 - np.pi / 2
    camera_vec = np.array(
        [np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)]
    )
    line, norm = torus_line(500, r, R, u, v)
    cond = np.dot(norm, camera_vec) >= 0
    bounds = get_obscure_segs(cond)

    # Did not account for normal vectors intersecting with main body, so
    # we have manual overrides...
    print(bounds)
    bounds[3] = 220
    print(bounds)

    for i in range(len(bounds) - 1):
        seg = line[bounds[i] : bounds[i + 1] + 1, :]
        lx, ly, lz = seg[:, 0], seg[:, 1], seg[:, 2]
        style = "solid" if cond[bounds[i]] else "dashed"
        if i == 3:
            style = "dashed"
        if i == 4:
            style = "dashed"
        ax.plot(lx, ly, lz, color="blue", linewidth=2, linestyle=style)

    # Display points.
    ax.scatter(*torus_points(r, R, u, v, k), color="purple", s=50)
    # ax.plot(*torus_line(200, 2, 1, 2, 1), color="orange", linewidth=3)
    plt.show()
    # plt.savefig(f"Torus - N{k}({u},{v}).png", bbox_inches="tight")


if __name__ == "__main__":
    main()
