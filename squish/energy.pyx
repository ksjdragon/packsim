import numpy as np
from cython.parallel import parallel, prange

cimport numpy as np
from libc.math cimport NAN, pi as PI, atanh
from squish.core cimport INT_T, FLOAT_T, Vector2D, Matrix2x2, BitSet, \
    _Vector2D, _Matrix2x2, _BitSet
from squish.voronoi cimport Site, HalfEdge, EdgeCacheMap, VoronoiInfo, \
    _Site, _HalfEdge, _EdgeCacheMap, _VoronoiInfo, VoronoiContainer, \
    R, NAN_MATRIX, NAN_VECTOR

#### Constants ####

INT = np.int64
FLOAT = np.float64

cdef EdgeCacheMap AREA_ECM = _EdgeCacheMap(0, 4, 6, 8, 10, 12, 13, -1, -1, 14)
cdef EdgeCacheMap RADIALT_ECM = _EdgeCacheMap(0, 4, 6, 8, -1, 10, 11, 12, 13, 14)

cdef class AreaEnergy(VoronoiContainer):
    """
    Class for formulas relevant to the Area energy.
    :param n: [int] how many sites to generate.
    :param w: [float] width of the bounding domain.
    :param h: [float] height of the bounding domain.
    :param r: [float] radius of zero energy circle.
    :param sites: [np.ndarray] collection of sites.
    """

    attr_str = "area"
    title_str = "Area"

    def __init__(AreaEnergy self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
                    np.ndarray[FLOAT_T, ndim=2] site_arr):
        self.edge_cache_map = &AREA_ECM
        self.energy = 0.0

        super().__init__(n, w, h, r, site_arr)
        self.minimum = (<FLOAT_T>n)*(w*h/(<FLOAT_T>n)-PI*r**2)**2


    cdef void precompute(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
            self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        cdef Site xi
        cdef HalfEdge em, e, ep
        cdef Vector2D vdiff

        cdef FLOAT_T [:] site_energy = np.full(self.sites.shape[0], PI*self.r**2)

        cdef INT_T i, j
        for i in prange(self.sites.shape[0], nogil=True):
            xi = _Site(i, &info)
            e = xi.edge(&xi)

            site_energy[i] = (xi.cache.area(&xi, NAN) - site_energy[i])**2
            xi.cache.energy(&xi, site_energy[i])

            j = 0
            while j < xi.edge_num(&xi):
                em, ep = e.prev(&e), e.next(&e)
                vdiff = em.origin(&em)
                vdiff.self.vsub(&vdiff, ep.origin(&ep))
                e.cache.dVdv(&e, R.vecmul(&R, vdiff))
                e.cache.H(&e, VoronoiContainer.calc_H(em, e))

                e = e.next(&e)
                j = j + 1

        self.energy = np.sum(site_energy[:self.n])


    cdef void calc_grad(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
                self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        cdef Site xi, xf
        cdef HalfEdge e, f
        cdef Vector2D dedxi_p
        cdef BitSet edge_set

        cdef INT_T num_edges = self.edges.shape[0]
        cdef FLOAT_T A = PI*self.r**2

        cdef FLOAT_T [:, ::1] dedx = np.zeros((self.n, 2), dtype=FLOAT)

        cdef INT_T i, j
        for i in prange(self.n, nogil=True):
            xi = _Site(i, &info)
            e = xi.edge(&xi)
            edge_set = _BitSet(num_edges)
            j = 0
            while j < xi.edge_num(&xi): # Looping through site edges.
                f = e
                while True: # Circling this vertex.
                    if not edge_set.add(&edge_set, f.arr_index):
                        xf = f.face(&f)
                        dedxi_p = f.cache.dVdv(&f, NAN_VECTOR)  #dVdv
                        dedxi_p.self.smul(&dedxi_p, xf.cache.area(&xf, NAN) - A)
                        dedxi_p.self.matmul(&dedxi_p, e.cache.H(&e, NAN_MATRIX))
                        dedx[i][0] += dedxi_p.x
                        dedx[i][1] += dedxi_p.y

                    f = f.twin(&f)
                    f = f.next(&f)
                    if f.arr_index == e.arr_index:
                        break

                e = e.next(&e)
                j = j + 1
            edge_set.free(&edge_set)
        self.grad = dedx


    cdef void calc_hess(self) except *:
        d = 10e-5
        HE = np.zeros((2*self.n, 2*self.n))
        new_sites = np.copy(self.site_arr)    # Maintain one copy for speed.
        for i in range(self.n):
            for j in range(2):
                mod = self.w if j == 0 else self.h
                new_sites[i][j] = (new_sites[i][j] + d) % mod
                Ep = self.__class__(self.n, self.w, self.h, self.r, new_sites)
                new_sites[i][j] = (new_sites[i][j] - 2*d) % mod
                Em = self.__class__(self.n, self.w, self.h, self.r, new_sites)
                new_sites[i][j] = (new_sites[i][j] + d) % mod

                HE[:, 2*i+j] = ((Ep.gradient - Em.gradient)/(2*d)).flatten()

        # Average out discrepencies, since it should be symmetric.
        for i in range(2*self.n):
            for j in range(i, 2*self.n):
                HE[i][j] = (HE[i][j] + HE[j][i])/2
                HE[j][i] = HE[i][j]
        self.hess = HE


cdef class RadialALEnergy(VoronoiContainer):
    """
    Class for formulas relevant to the Area energy.
    :param n: [int] how many sites to generate.
    :param w: [float] width of the bounding domain.
    :param h: [float] height of the bounding domain.
    :param r: [float] radius of zero energy circle.
    :param sites: [np.ndarray] collection of sites.
    """

    attr_str = "radial-al"
    title_str = "Radial[AL]"


    def __init__(AreaEnergy self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
                    np.ndarray[FLOAT_T, ndim=2] site_arr):
        #self.edge_cache_map = &AREA_EDGE_CACHE_MAP
        self.energy = 0.0

        super().__init__(n, w, h, r, site_arr)


    cdef void precompute(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
            self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        pass


    cdef void calc_grad(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
                self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        pass

    cdef void calc_hess(self) except *:
        pass


cdef class RadialTEnergy(VoronoiContainer):
    """
    Class for formulas relevant to the Area energy.
    :param n: [int] how many sites to generate.
    :param w: [float] width of the bounding domain.
    :param h: [float] height of the bounding domain.
    :param r: [float] radius of zero energy circle.
    :param sites: [np.ndarray] collection of sites.
    """

    attr_str = "radial-t"
    title_str = "Radial[T]"
    def __init__(AreaEnergy self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
                    np.ndarray[FLOAT_T, ndim=2] site_arr):
        self.edge_cache_map = &RADIALT_ECM
        self.energy = 0.0

        super().__init__(n, w, h, r, site_arr)


    cdef void precompute(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
            self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        cdef Site xi
        cdef HalfEdge e, ep
        cdef Vector2D la

        # All energy has a 2pir_0 term.
        cdef FLOAT_T [:] site_energy = np.full(self.sites.shape[0], 2*PI*self.r**2)
        cdef FLOAT_T [:] avg_radii = np.zeros(self.sites.shape[0])
        cdef FLOAT_T sm, sp

        cdef INT_T i, j
        for i in prange(self.sites.shape[0], nogil=True):
            xi = _Site(i, &info)
            e = xi.edge(&xi)
            j = 0
            while j < xi.edge_num(&xi):
                ep = e.next(&e)
                #e.cache.H(&e, VoronoiContainer.calc_H(em, e))

                la = e.cache.la(&e, NAN_VECTOR)
                sp = la.dot(&la, ep.cache.da(&ep, NAN_VECTOR)) # dap . la
                sm = la.dot(&la, e.cache.da(&e, NAN_VECTOR)) # da . la

                sp = sp / (ep.cache.da_mag(&ep, NAN) * e.cache.la_mag(&e, NAN))
                sm = sm / (e.cache.da_mag(&e, NAN) * e.cache.la_mag(&e, NAN))

                e.cache.calI(&e, <FLOAT_T>(atanh(<double>sp) - atanh(<double>sm)))

                avg_radii[i] += e.cache.ya_mag(&e, NAN) * e.cache.calI(&e, NAN) / 2

                e = e.next(&e)
                j = j + 1

            site_energy[i] += 2*(xi.cache.area(&xi, NAN) - self.r*avg_radii[i])

            xi.cache.avg_radius(&xi, avg_radii[i]/(2*PI))
            xi.cache.energy(&xi, site_energy[i])

        self.energy = np.sum(site_energy[:self.n])


    cdef void calc_grad(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
                self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        cdef Site xi
        cdef HalfEdge e
        cdef Vector2D dedxi_p

        cdef FLOAT_T [:, ::1] dedx = np.zeros((self.n, 2), dtype=FLOAT)

        cdef INT_T i, j
        for i in prange(self.n, nogil=True):
            xi = _Site(i, &info)
            e = xi.edge(&xi)

            j = 0
            while j < xi.edge_num(&xi): # Looping through site edges.
                dedxi_p = e.cache.ya(&e, NAN_VECTOR)
                dedxi_p.self.smul(
                    &dedxi_p,
                    e.cache.calI(&e, NAN) / e.cache.ya_mag(&e, NAN)
                )

                dedx[i][0] += 2*self.r*dedxi_p.x
                dedx[i][1] += 2*self.r*dedxi_p.y

                e = e.next(&e)
                j = j + 1

        self.grad = dedx

    cdef void calc_hess(self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
                                             self.vertices, self.site_cache,
                                             self.edge_cache, self.edge_cache_map)
        cdef Site xi, xk
        cdef HalfEdge em, e, ep, f
        cdef Vector2D temp1, temp2
        cdef Matrix2x2 dsite, q, tempm
        cdef BitSet edge_set

        cdef INT_T num_edges = self.edges.shape[0]

        cdef FLOAT_T [:,:] HE = np.zeros((2*self.n, 2*self.n))

        cdef INT_T i, j, k
        for i in prange(self.sites.shape[0], nogil=True):
            xi = _Site(i, &info)
            e = xi.edge(&xi)

            j = 0
            while j < xi.edge_num(&xi):
                em, ep = e.prev(&e), e.next(&e)
                e.cache.H(&e, VoronoiContainer.calc_H(em, e))

                e = e.next(&e)
                j = j + 1


        for i in range(self.n):
            xi = _Site(i, &info)
            e = xi.edge(&xi)
            edge_set = _BitSet(num_edges)

            j = 0
            while j < xi.edge_num(&xi):
                em, ep = e.prev(&e), e.next(&e)

                # Calculating of p
                temp1 = em.cache.la(&em, NAN_VECTOR)
                temp1.self.rot(&temp1)
                temp1.self.sdiv(
                    &temp1,
                    em.cache.la_mag(&em, NAN) * em.cache.ya_mag(&em, NAN) / 2
                )
                temp2 = e.cache.la(&e, NAN_VECTOR)
                temp2.self.rot(&temp2)
                temp2.self.sdiv(
                    &temp2,
                    e.cache.la_mag(&e, NAN) * e.cache.ya_mag(&e, NAN) / 2
                )

                temp1.self.vsub(&temp1, temp2)  # (lm/Am - l/A)

                temp2 = e.cache.da(&e, NAN_VECTOR) # rot(d) / |d|
                temp2.self.sdiv(&temp2, e.cache.da_mag(&e, NAN))
                temp2.self.rot(&temp2)

                dsite = temp1.vecmul(&temp1, temp2)

                HE[2*i, 2*i] -= dsite.a
                HE[2*i, 2*i+1] -= dsite.b
                HE[2*i+1, 2*i] -= dsite.c
                HE[2*i+1, 2*i+1] -= dsite.d

                # Calculating of q
                temp2 = e.cache.la(&e, NAN_VECTOR)
                temp1 = temp2.copy.rot(&temp2)
                q = temp1.vecmul(&temp1, temp2)
                q.self.sdiv(&q, e.cache.la_mag(&e, NAN)**2)

                q.self.msub(&q, R)
                q.self.smul(
                    &q,
                    e.cache.calI(&e, NAN) / e.cache.la_mag(&e, NAN)
                )

                temp1 = e.cache.la(&e, NAN_VECTOR)
                temp1.self.rot(&temp1)
                tempm = temp1.vecmul(&temp1, temp1)
                tempm.self.smul(
                    &tempm,
                    ep.cache.da_mag(&ep, NAN) - e.cache.da_mag(&e, NAN)
                )
                tempm.self.sdiv(
                    &tempm,
                    e.cache.la_mag(&e, NAN)**3 * e.cache.ya_mag(&e, NAN) / 2
                )

                q.self.madd(&q, tempm)

                # Minus portion
                temp2 = em.cache.la(&em, NAN_VECTOR)
                temp1 = temp2.copy.rot(&temp2)
                tempm = temp1.vecmul(&temp1, temp2)
                tempm.self.sdiv(&tempm, em.cache.la_mag(&em, NAN)**2)

                tempm = R.copy.msub(&R, tempm)
                tempm.self.smul(
                    &tempm,
                    em.cache.calI(&em, NAN) / em.cache.la_mag(&em, NAN)
                )

                q.self.madd(&q, tempm)

                temp1 = em.cache.la(&em, NAN_VECTOR)
                temp1.self.rot(&temp1)
                tempm = temp1.vecmul(&temp1, temp1)
                tempm.self.smul(
                    &tempm,
                    e.cache.da_mag(&e, NAN) - em.cache.da_mag(&em, NAN)
                )
                tempm.self.sdiv(
                    &tempm,
                    em.cache.la_mag(&em, NAN)**3 * em.cache.ya_mag(&em, NAN) / 2
                )

                q.self.msub(&q, tempm)

                # Add p to q, so p = p, q = p+q
                q.self.madd(&q, dsite)

                f = e
                while True:
                    xk = f.face(&f)
                    k = xk.index(&xk) % self.n
                    if k < 0:
                        k = k + self.n

                    if not edge_set.add(&edge_set, f.arr_index):
                        tempm = q.copy.matmul(&q, f.cache.H(&f, NAN_MATRIX))

                        HE[2*i, 2*k] += tempm.a
                        HE[2*i, 2*k+1] += tempm.b
                        HE[2*i+1, 2*k] += tempm.c
                        HE[2*i+1, 2*k+1] += tempm.d

                    f = f.twin(&f)
                    f = f.next(&f)
                    if f.arr_index == e.arr_index:
                        break

                e = e.next(&e)
                j = j + 1
            edge_set.free(&edge_set)
        self.hess = -2*self.r*np.asarray(HE, dtype=FLOAT)
        self.hess = (
            ( np.asarray(self.hess, dtype=FLOAT)
            + np.asarray(self.hess, dtype=FLOAT).T )
            / 2
        )
