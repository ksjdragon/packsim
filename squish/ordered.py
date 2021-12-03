from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.linalg import norm as mag
from math import gcd, sqrt, log, tan, atan, pi

Config = Tuple[int, int]


def configurations(domain: DomainParams) -> List[Config]:
    n = domain.n
    coprimes, valid = [], []
    for i in range(n):
        for j in range(i):
            if gcd(i, j) == 1:
                coprimes.append((j, i))

    coprimes = set(coprimes)
    while len(coprimes) > 0:
        first = coprimes.pop()
        valid.append(first)
        for i in range(2, n):
            try:
                coprimes.remove(((first[0] * i) % n, (first[1] * i) % n))
            except KeyError:
                pass

    return valid


def get_config_generators(
    domain: DomainParams, config: Config
) -> Tuple[Config, Config]:
    n, w, h = domain.n, domain.w, domain.h
    q1 = sites(domain, config)[1:]
    all_sites = np.concatenate(
        (q1, q1 - np.array([w, 0]), q1 - np.array([0, h]), q1 - domain.dim)
    )
    # Sort sites by magnitude and smallest.
    all_sites = sorted(list(all_sites), key=lambda x: np.linalg.norm(x))
    v = all_sites[0]  # Smallest vector set to v.

    all_sites = np.array(all_sites[1:])  # Remove v from search set.
    # Checking 0 < ax + by < v*v to make the sites are within the region.
    tol = 1e-3
    vdot = np.matmul(all_sites, v)
    in_box = all_sites[np.where((-tol <= vdot) & (vdot <= (v.dot(v) + tol)))[0]]
    in_box = np.expand_dims(in_box, 0).swapaxes(
        0, 1
    )  # Used for the next step, getting site*site

    v2 = in_box[
        np.argmin(np.squeeze(np.matmul(in_box, in_box.transpose(0, 2, 1))))
    ].flatten()

    if np.all(v == v2):
        print(v, v2, n, w, h, config)
    return tuple(v), tuple(v2)


def sites(domain: DomainParams, config: Config) -> numpy.ndarray:
    n, w, h = domain.n, domain.w, domain.h
    config, mults = np.array(config), np.arange(domain.n)
    return (config * np.dstack((w * mults, h * mults))[0] / n) % domain.dim


def area(domain: DomainParams, config: Config) -> float:
    v, w = get_config_generators(domain, config)
    v, w = np.array(v), np.array(w)
    c = circumcenter(v, w)

    return (
        mag(v) * mag(v / 2 - c)
        + mag(w) * mag(w / 2 - c)
        + mag(v - w) * mag((v + w) / 2 - c)
    )


def avg_radius(domain: DomainParams, config: Config) -> float:
    v, w = get_config_generators(domain, config)
    v, w = np.array(v), np.array(w)
    c = circumcenter(v, w)

    return 2 * (
        avg_rp(mag(v), 2 * mag(v / 2 - c))
        + avg_rp(mag(w), 2 * mag(w / 2 - c))
        + avg_rp(mag(v - w), 2 * mag((v + w) / 2 - c))
    )


def avg_rp(d: float, l: float) -> float:
    return (d / (4 * pi)) * log(tan(0.5 * (atan(l / d) + pi / 2)) ** 2)


def circumcenter(v: numpy.ndarray, w: numpy.ndarray) -> numpy.ndarray:
    det = 1 / (2 * rot(v).dot(w))
    if rot(v).dot(w) == 0:
        print(v, w)
    v2, w2 = v.dot(v), w.dot(w)
    c = np.empty((2,))
    c[0], c[1] = w[1] * v2 - v[1] * w2, -w[0] * v2 + v[0] * w2
    return det * c


def rot(v: numpy.ndarray) -> numpy.ndarray:
    w = np.copy(v)
    w[0], w[1] = -w[1], w[0]
    return w
