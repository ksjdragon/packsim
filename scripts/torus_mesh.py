import argparse, numpy as np, os
from stl.stl import ASCII
from stl import mesh
from squish import DomainParams, Simulation, ordered
from squish.common import OUTPUT_DIR
from scipy.spatial import Voronoi

SUBDIVISION_AMOUNT = 8


def not_in_order(i, j):
    if i < j:
        return j - i != 1
    else:
        return i - j == 1


def centroid(site, verts):
    area, c = 0, 0
    v = verts - site
    # print(v)
    for i in range(len(v)):
        x, y = v[i], v[(i + 1) % len(v)]
        a = np.cross(x, y)
        area += a
        c += a * (x + y)

    return c / (3 * area) + site


def flat_sheet(k):
    x = np.linspace(-np.pi, np.pi, 101)
    x = np.transpose([np.tile(a, len(a)), np.repeat(a, len(a))])
    all_verts = np.hstack((a, np.zeros((len(a), 1))))

    xx = np.array([[x for x in range(101 * 100) if (x - 100) % 101 > 0]]).T

    b = np.hstack((xx, xx + 1, xx + 101))
    c = np.hstack((xx + 1, xx + 102, xx + 101))
    all_faces = np.concatenate((b, c))

    return all_verts, all_faces


def torus_transform(v, R, r):
    new_verts = np.empty(v.shape)
    # Rotate by 90 degrees, and then shift to range [0, 2\pi]
    v[:, :2] = np.matmul(np.array([[0, -1], [1, 0]]), v[:, :2].T).T
    v[:, :2] = v[:, :2] % (2 * np.pi)
    # v[:, 0], v[:, 1] = v[:, 1], v[:, 0]

    new_verts[:, 0] = (R + r * np.cos(v[:, 1])) * np.cos(v[:, 0])
    new_verts[:, 1] = (R + r * np.cos(v[:, 1])) * np.sin(v[:, 0])
    new_verts[:, 2] = r * np.sin(v[:, 1])
    return new_verts


def flat_region(c, verts, height):
    m = len(verts)

    centers = np.hstack((np.array([c, c]), np.array([[height], [0]])))
    v_top = np.hstack((verts, height * np.ones((m, 1))))
    v_bot = np.hstack((verts, np.zeros((m, 1))))

    v = np.concatenate((centers, v_top, v_bot))

    vi = np.atleast_2d(np.arange(m, dtype=int)).T
    cent_top, cent_bot = np.zeros((m, 1), dtype=int), np.ones((m, 1), dtype=int)
    vert_top_m, vert_top_p = vi + 2, (vi + 1) % m + 2
    vert_bot_m, vert_bot_p = vi + m + 2, (vi + 1) % m + m + 2

    top_face = np.hstack((cent_top, vert_top_m, vert_top_p))
    bot_face = np.hstack((cent_bot, vert_bot_p, vert_bot_m))

    f = np.concatenate((top_face, bot_face))
    return v, f, 2


def torus_region(c, verts, height):
    sd = SUBDIVISION_AMOUNT
    d = verts - c
    m = len(verts)
    # 1 + 6 + 12 ... verts, 1 + 3 + 5 ... faces.
    v, f = (
        np.empty((1 + m * sd * (sd - 1) // 2, 2)),
        np.empty((m * (sd - 1) ** 2, 3), dtype=int),
    )
    v[0] *= 0

    # Vertices are ordered by rings going out from the center.
    v_inds = np.arange(-1, sd)
    v_inds = 1 + m * v_inds * (v_inds + 1) // 2
    v_inds[0] = 0

    for i in range(1, sd):
        fj, fk = m * (i - 1) ** 2, m * i ** 2
        region_verts = [
            np.linspace(i * d[j] / (sd - 1), i * d[(j + 1) % m] / (sd - 1), i + 1)[:-1]
            for j in range(m)
        ]
        v[v_inds[i] : v_inds[i + 1]] = np.concatenate(region_verts)

        faces = []
        for j in range(m):
            # v_off_inds = v_inds + j * np.arange(sd + 1)
            r1, r2 = (
                (j * (i - 1) + np.arange(i + 1)) % (m * (i - 1)) + v_inds[i - 1],
                (j * i + np.arange(i + 2)) % (m * i) + v_inds[i],
            )
            r1, r2 = np.atleast_2d(r1).T, np.atleast_2d(r2).T
            faces.append(np.hstack((r1[:-1], r2[:-2], r2[1:-1])))
            faces.append(np.hstack((r1[:-2], r2[1:-2], r1[1:-1])))

        f[fj:fk] = np.concatenate(faces)

    v += c
    v = np.hstack((v, height * np.ones((len(v), 1))))
    return v, f, v_inds[sd - 1]


def main():
    parser = argparse.ArgumentParser(description="Generates 3D model for equilibrium.")
    parser.add_argument(
        "sims_path", metavar="sim_dir", help="simulation to obtain equilibrium from"
    )
    parser.add_argument("frame_num", metavar="FRAME_NUM", type=int, help="frame number")
    parser.add_argument("size", metavar="SIZE", type=int, help="width in mm")
    parser.add_argument(
        "--torus", dest="torus", action="store_true", help="generates torus model"
    )
    args = parser.parse_args()

    torus = args.torus
    size = args.size * 10

    # Get desired frame and load.
    sim, frames = Simulation.load(args.sims_path)
    frames = list(frames)
    # num = 100
    # nums = [838. 348. 14. 664, 725, 974]
    # frames.sort(key=lambda x: x["energy"])
    # frame = frames[-32]
    # for i, f in enumerate(frames):
    #    if i == 858:
    frame = frames[args.frame_num]
    n, w, h = frame["domain"][0], frame["domain"][1], frame["domain"][2]
    frame = sim.energy.mode(*frame["domain"], frame["arr"])

    # Set up size and scaling variables.
    dim = np.array([w, h])
    vor = frame.vor_data

    alpha = w / h
    if torus:
        r = alpha * size / 2
        R = size / 2 - r
        height_bounds = (0.5 * r, 1.25 * r)
        scale = np.array([2 * np.pi, 2 * np.pi])
    else:
        height_bounds = (size / 10, 0.75 * size)
        scale = np.array([alpha * size, size])

    # Rescale and translate domain.
    points = (vor.points - dim / 2) * scale / dim
    vertices = (vor.vertices - dim / 2) * scale / dim

    # Obtain height scaling.
    heights = frame.stats["site_energies"] - ordered.e_hex(sim.domain)
    heights -= np.min(heights)
    heights *= (height_bounds[1] - height_bounds[0]) / np.max(heights)
    heights += height_bounds[0]

    # Prepare oriented regions and site_vert list
    sites = points[:n]
    regions = []
    for i, x in enumerate(vor.point_region[:n]):
        region = vor.regions[x]
        x = sites[i]
        p, q = vertices[region[0]] - x, vertices[region[1]] - x
        regions.append(region if np.cross(p, q) >= 0 else region[::-1])

    site_verts = []
    for region in regions:
        site_verts.append(vertices[region])

    all_verts, all_faces = [], []
    offsets = np.empty((n,), dtype=int)
    for i, verts in enumerate(site_verts):
        c = centroid(sites[i], verts)
        if torus:
            face_verts, faces, offset = torus_region(c, verts, heights[i])
        else:
            face_verts, faces, offset = flat_region(c, verts, heights[i])

        old_len = len(all_verts)
        if i == 0:
            all_verts, all_faces = face_verts, faces
        else:
            all_verts = np.concatenate((all_verts, face_verts))
            all_faces = np.concatenate((all_faces, faces + old_len))

        offsets[i] = old_len + offset

    # Merge regions
    for i, x in enumerate(vor.ridge_points):
        x, y = x[0], x[1]  # Site indices.
        no_x, no_y = x >= sim.domain.n, y >= sim.domain.n
        if no_x and no_y:
            continue

        adj_verts = vor.ridge_vertices[i]
        v1, v2 = vertices[adj_verts[0]], vertices[adj_verts[1]]

        bot_face, top_face = None, None
        if torus:
            sd = SUBDIVISION_AMOUNT - 1
            if no_x:
                x, y = y, x
            y = y % n

            v1_x_ind = regions[x].index(adj_verts[0])
            v2_x_ind = regions[x].index(adj_verts[1])
            # Need to find matching vertices
            v1, v2 = v1 % (2 * np.pi), v2 % (2 * np.pi)
            y_verts = vertices[regions[y]] % (2 * np.pi)
            v1_y_ind = np.argmin(np.linalg.norm(y_verts - v1, axis=1))
            v2_y_ind = np.argmin(np.linalg.norm(y_verts - v2, axis=1))

            # We want x lower than y.
            off_x, off_y = offsets[x], offsets[y]
            if heights[x] > heights[y]:
                v1_x_ind, v1_y_ind = v1_y_ind, v1_x_ind
                v2_x_ind, v2_y_ind = v2_y_ind, v2_x_ind
                off_x, off_y = off_y, off_x

            if not_in_order(v1_x_ind, v2_x_ind):
                v1_x_ind, v2_x_ind = v2_x_ind, v1_x_ind

            if not_in_order(v1_y_ind, v2_y_ind):
                v1_y_ind, v2_y_ind = v2_y_ind, v1_y_ind

            sds = np.atleast_2d(np.arange(sd)).T
            xm, ym = sd * v1_x_ind + sds, sd * v1_y_ind + sds
            xp, yp = xm + 1, ym + 1
            xp[-1], yp[-1] = sd * v2_x_ind, sd * v2_y_ind

            xm += off_x
            xp += off_x
            ym += off_y
            yp += off_y

            bot_face = np.hstack((xm, ym[::-1], xp))
            top_face = np.hstack((xm, yp[::-1], ym[::-1]))

        else:
            if no_x ^ no_y:
                if no_x:
                    x = y
                m = len(regions[x])
                v1_ind = regions[x].index(adj_verts[0])
                v2_ind = regions[x].index(adj_verts[1])
                if not_in_order(v1_ind, v2_ind):
                    v1_ind, v2_ind = v2_ind, v1_ind

                v1_ind += offsets[x]
                v2_ind += offsets[x]
                bot_face = np.array([[v1_ind, m + v1_ind, m + v2_ind]], dtype=int)
                top_face = np.array([[v1_ind, m + v2_ind, v2_ind]], dtype=int)
            else:
                v1_x_ind = regions[x].index(adj_verts[0])
                v2_x_ind = regions[x].index(adj_verts[1])
                v1_y_ind = regions[y].index(adj_verts[0])
                v2_y_ind = regions[y].index(adj_verts[1])

                # We want x lower than y.
                off_x, off_y = offsets[x], offsets[y]
                if heights[x] > heights[y]:
                    v1_x_ind, v1_y_ind = v1_y_ind, v1_x_ind
                    v2_x_ind, v2_y_ind = v2_y_ind, v2_x_ind
                    off_x, off_y = off_y, off_x

                if not_in_order(v1_x_ind, v2_x_ind):
                    v1_x_ind, v2_x_ind = v2_x_ind, v1_x_ind

                if not_in_order(v1_y_ind, v2_y_ind):
                    v1_y_ind, v2_y_ind = v2_y_ind, v1_y_ind

                v1_x_ind += off_x
                v2_x_ind += off_x
                v1_y_ind += off_y
                v2_y_ind += off_y

                bot_face = np.array([[v1_x_ind, v1_y_ind, v2_x_ind]], dtype=int)
                top_face = np.array([[v1_x_ind, v2_y_ind, v1_y_ind]], dtype=int)

        all_faces = np.concatenate((all_faces, bot_face, top_face))

    if torus:
        all_verts = torus_transform(all_verts, R, all_verts[:, 2])

    # Output into mesh
    surf = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(all_faces):
        for j in range(3):
            surf.vectors[i][j] = all_verts[f[j], :]

    m_type = "Torus" if torus else "Flat"
    out = OUTPUT_DIR / f"N{n} - {m_type} - {args.frame_num}.stl"
    # temp = f"n{n}{m_type.lower()}{num}.stl"
    surf.save(out)
    # os.rename(temp, out)
    print(f"Wrote to {out}.")
    # n, alpha = 20, np.sqrt(3) * 2 / 5
    # u, v = 5, 2
    # domain = DomainParams(n, alpha, 1, 1)
    # size = 1

    # R, r = size, size * (1 - alpha) / alpha

    # sites = ordered.sites(domain, (u, v))


if __name__ == "__main__":
    np.seterr(divide="ignore")
    main()
