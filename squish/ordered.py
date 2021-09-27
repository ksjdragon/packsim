from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.linalg import norm as mag
from math import gcd, sqrt, log, tan, atan, pi

Config = Tuple[int, int]

def configurations(domain: DomainParams) -> List[Config]:
	n, w, h = domain.n, domain.w, domain.h
	valid = []
	mults = np.arange(n)
	configs = np.dstack((np.repeat(mults,n).T, np.tile(mults, n).T))[0][1:]
	for i in range(len(configs)):
		eq_x = n if configs[i][0] == 0 else configs[i][0]
		eq_y = n if configs[i][1] == 0 else configs[i][1]

		if gcd(eq_x, eq_y) != 1:
			continue

		vecs = configs[i]*np.dstack((w*mults, h*mults)).swapaxes(0,1)/n % domain.dim
		vmod2 = np.squeeze(np.matmul(vecs, vecs.transpose(0,2,1)))
		vmodv = np.squeeze(vecs).dot(vecs[1].T).T.flatten()

		if np.all(vmod2 >= vmodv):
			valid.append(tuple(configs[i]))

	return valid


def get_config_generators(domain: DomainParams, config: Config) -> Tuple[Config, Config]:
	n, w, h = domain.n, domain.w, domain.h
	q1 = sites(domain, config)
	v = q1[1]
	# Sites to check can ignore 0*v and v itself.
	all_sites = np.concatenate((q1, q1-[w,0], q1-[w,h], q1-[0,h]))[2:]

	# Checking 0 < ax + by < v*v to make the sites are within the region.
	tol = 1e-3
	vdot = np.matmul(all_sites, v)
	in_box = all_sites[np.where((-tol <= vdot) & (vdot <= (v.dot(v)+tol)))[0]]
	in_box = np.expand_dims(in_box, 0).swapaxes(0,1)	# Used for the next step, getting site*site

	w = in_box[np.argmin(np.squeeze(np.matmul(in_box, in_box.transpose(0,2,1))))].flatten()

	return tuple(v), tuple(w)


def sites(domain: DomainParams, config: Config) -> numpy.ndarray:
	n, w, h = domain.n, domain.w, domain.h
	config, mults = np.array(config), np.arange(domain.n)
	return (config*np.dstack((w*mults, h*mults))[0]/n) % domain.dim


def area(domain: DomainParams, config: Config) -> float:
	v, w = get_config_generators(domain, config)
	v, w = np.array(v), np.array(w)
	c = circumcenter(v, w)

	return mag(v)*mag(v/2 - c) + mag(w)*mag(w/2-c) + mag(v-w)*mag((v+w)/2-c)


def avg_radius(domain: DomainParams, config: Config) -> float:
	v, w = get_config_generators(domain, config)
	v, w = np.array(v), np.array(w)
	c = circumcenter(v, w)

	return 2*(avg_rp(mag(v), 2*mag(v/2 - c)) + avg_rp(mag(w), 2*mag(w/2-c)) + \
			 avg_rp(mag(v-w),2*mag((v+w)/2-c)))


def avg_rp(d: float, l: float) -> float:
	return (d/(4*pi))*log(tan(.5*(atan(l/d)+pi/2))**2)


def circumcenter(v: numpy.ndarray, w: numpy.ndarray) -> Config:
	det = 1/(2*rot(v).dot(w))
	v2, w2 = v.dot(v), w.dot(w)
	c = np.empty((2,))
	c[0], c[1] = w[1]*v2 - v[1]*w2,-w[0]*v2 + v[0]*w2
	return det*c


def rot(v: numpy.ndarray) -> numpy.ndarray:
	w = np.copy(v)
	w[0], w[1] = -w[1], w[0]
	return w