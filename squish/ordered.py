from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.linalg import norm as mag
from math import gcd, sqrt, log, tan, atan, atanh, pi
import cmath
from fractions import Fraction

Config = Tuple[int, int]
R_HEX = (6 * 3 ** (-0.25) * sqrt(2) * atanh(0.5)) / (2 * pi)


def e_hex(domain: DomainParams) -> float:
    return 2 - 2 * domain.r * (2 * pi) * R_HEX + 2 * pi * domain.r ** 2


def toroidal_distance(domain: DomainParams, x: np.ndarray, y: np.ndarray) -> float:
    dim = np.array([domain.w, domain.h])
    absdist = np.abs(y - x)
    wrap = dim * (absdist >= dim / 2)
    return np.linalg.norm(wrap - absdist, axis=1)


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

    for i in range(len(valid)):
        valid.append((valid[i][1], valid[i][0]))

    return valid


def get_config_generators(
    domain: DomainParams, config: Config
) -> Tuple[Config, Config]:
    # x = sites(domain, config)
    # x = x[x[:, 1] <= (domain.h / 2)]
    # mag_x = toroidal_distance(domain, x, np.zeros(x.shape))
    # x = x[np.argsort(mag_x)]
    # print(x)

    # return [tuple(x[1]), tuple(x[2])]

    n, w, h = domain.n, domain.w, domain.h
    x = sites(domain, config)[1:]  # Remove 0
    # Concatenate quadrants 1, 2 and 4.
    x = np.concatenate(
        (x, x - np.array([w, 0]), x - np.array([0, h]), x - np.array([w, h]))
    )

    # Reduce search space
    # x = x[(np.abs(x[:, 0]) <= (domain.w / 2)) * (np.abs(x[:, 1]) <= (domain.h / 2))]
    mag_x = np.linalg.norm(x, axis=1)
    x = x[np.argsort(mag_x)]  # Sort by magnitude

    v = x[0]
    x = x[1:]  # Remove v from search set.
    # Checking 0 < ax + by < v*v to make the sites are within the region.
    tol = 1e-8
    vdot = np.matmul(x, v)
    x = x[(-tol <= vdot) * (vdot <= v.dot(v) + tol)]
    mag_x = np.linalg.norm(x, axis=1)
    x = x[np.argsort(mag_x)]

    v2 = x[0]

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
        pass
        # print(v, w)
    v2, w2 = v.dot(v), w.dot(w)
    c = np.empty((2,))
    c[0], c[1] = w[1] * v2 - v[1] * w2, -w[0] * v2 + v[0] * w2
    return det * c


def rot(v: numpy.ndarray) -> numpy.ndarray:
    w = np.copy(v)
    w[0], w[1] = -w[1], w[0]
    return w


def divisors(n: int) -> List[int]:
    divs = [[i, n // i] for i in range(1, int(sqrt(n)) + 1) if n % i == 0]
    return sorted(set(list(sum(divs, []))))


def factorize(n: int) -> Dict[int, int]:
    primes = [i for i in range(1, n + 1) if len(divisors(i)) == 2]
    prime_fac = {}
    for prime in primes:
        i = 0
        while n % prime ** (i + 1) == 0:
            i += 1
        if i > 0:
            prime_fac[prime] = i

    return prime_fac


def hexagon_alpha(n: int, fraction: bool = False) -> List[int]:
    if n % 2 == 1:
        return []

    fac = factorize(n)
    q = 1
    for prime in fac:
        if prime % 3 != 2:
            q *= prime ** fac[prime]

    divq = divisors(q)
    ratios, thres = [], 1 / sqrt(3)
    for g in divq:
        us = n // (2 * g)
        divu = divisors(us)
        for u in divu:
            d = 2 * g * u * u
            f = Fraction(n, d)
            if f <= thres:
                ratios.append(f)
            else:
                ratios.append(Fraction(d, 3 * n))

    ratios = sorted(set(ratios))
    if fraction:
        return ratios
    else:
        return [float(x) * sqrt(3) for x in ratios]


def hexagon_alpha_brute(n: int):
    w = cmath.rect(1, 2 * pi / 3)
    divs = divisors(n / 2)
    ratios, thres = [], 1 / sqrt(3)
    for a in range(n):
        for b in range(1, a + 1):
            if a == 0 and b == 0:
                continue
            z2, g = a * a - a * b + b * b, gcd(a - 2 * b, 2 * a - b)
            if z2 // g in divs:
                f = Fraction(n, 2 * z2)
                if f <= thres:
                    ratios.append(f)
                else:
                    ratios.append(Fraction(2 * z2, 3 * n))

    ratios = sorted(set([float(x) * sqrt(3) for x in ratios]))
    return ratios
