import numpy as np
import matplotlib.pyplot as plt


def torus_mesh(n, c, a):
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    return x, y, z


def torus_line(n, c, a, u, v):
    phi = np.linspace(0, 2 * np.pi * u, n)
    theta = (v / u) * phi
    theta, phi = theta % (2 * np.pi), phi % (2 * np.pi)

    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    line = np.column_stack([x, y, z])
    x, y, z = c * np.cos(phi), c * np.sin(phi), 0 * phi
    norm = line - np.column_stack([x, y, z])

    return line, norm


def torus_points(c, a, u, v, k):
    phi = np.linspace(0, 2 * np.pi * u, k + 1)
    theta = (v / u) * phi
    theta, phi = theta % (2 * np.pi), phi % (2 * np.pi)

    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

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
    ax.set_zlim(-2, 2)
    ax.set_axis_off()

    azim, elev = 0, 40
    c, a = 2, 1
    u, v = 4, 1
    k = 10

    ax.view_init(elev, azim)

    ax.plot_surface(
        *torus_mesh(100, c, a), rcount=200, ccount=200, color="white", alpha=0.4
    )

    # Calculate when surface is facing or away and apply dotted vs solid lines.
    az, el = azim * np.pi / 180 - np.pi, elev * np.pi / 180 - np.pi / 2
    camera_vec = np.array(
        [np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)]
    )
    line, norm = torus_line(500, c, a, u, v)
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
    ax.scatter(*torus_points(c, a, u, v, k), color="purple", s=50)
    # ax.plot(*torus_line(200, 2, 1, 2, 1), color="orange", linewidth=3)
    plt.savefig(f"Torus - N{k}({u},{v}).png", bbox_inches="tight")


if __name__ == "__main__":
    main()
