import array, scipy.spatial, numpy as np
from cython.parallel import parallel, prange

cimport numpy as np
from cpython cimport array
from libc.math cimport isnan, NAN, pi as PI

from squish.core cimport INT_T, FLOAT_T, \
    IArray, FArray, Vector2D, Matrix2x2, \
    _IArray, _FArray, _Vector2D, _Matrix2x2

from squish.voronoi cimport SiteCacheMap, EdgeCacheMap, VoronoiInfo, Site, HalfEdge

#### Constants ####

INT = np.int64
FLOAT = np.float64

cdef Vector2D NAN_VECTOR = _Vector2D(NAN, NAN)
cdef Matrix2x2 NAN_MATRIX = _Matrix2x2(NAN, NAN, NAN, NAN)

cdef FLOAT_T[18] SYMM = [0,0, 1,0, 1,1, 0,1, -1,1, -1,0, -1,-1, 0,-1, 1,-1]
cdef Matrix2x2 R = _Matrix2x2(0, -1, 1, 0)

cdef SiteCacheMap SITE_CACHE_MAP = _SiteCacheMap(0, 1, 2, 3, 4, 5, -1)

#### SiteCacheMap Methods ####

cdef inline SiteCacheMap _SiteCacheMap(INT_T iarea, INT_T iperim, INT_T iisoparam,
                                       INT_T ienergy, INT_T iavg_radius,
                                       INT_T icentroid, INT_T imaxcenter) nogil:
    cdef SiteCacheMap sc
    sc.iarea, sc.iperim, sc.iisoparam, sc.ienergy, sc.iavg_radius = (
        iarea, iperim, iisoparam, ienergy, iavg_radius
    )
    sc.icentroid, sc.imaxcenter = icentroid, imaxcenter

    sc.area, sc.perim, sc.isoparam, sc.energy, sc.avg_radius = (
        area, perim, isoparam, energy, avg_radius
    )
    sc.centroid, sc.maxcenter = centroid, maxcenter

    return sc

cdef inline FLOAT_T area(Site* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.site_cache.get(&self.info.site_cache,
            (self.arr_index, self.cache.iarea)
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.iarea), val)
        return val

cdef inline FLOAT_T perim(Site* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.site_cache.get(&self.info.site_cache,
            (self.arr_index, self.cache.iperim)
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.iperim), val)
        return val

cdef inline FLOAT_T isoparam(Site* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.site_cache.get(&self.info.site_cache,
            (self.arr_index, self.cache.iisoparam)
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.iisoparam), val)
        return val

cdef inline FLOAT_T energy(Site* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.site_cache.get(&self.info.site_cache,
            (self.arr_index, self.cache.ienergy)
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.ienergy), val)
        return val

cdef inline FLOAT_T avg_radius(Site* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.site_cache.get(&self.info.site_cache,
            (self.arr_index, self.cache.iavg_radius)
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.iavg_radius), val)
        return val

cdef inline Vector2D centroid(Site* self, Vector2D val) nogil:
    if isnan(<double>val.x):
        return _Vector2D(
            self.info.site_cache.get(&self.info.site_cache,
                (self.arr_index, self.cache.icentroid)
            ),
            self.info.site_cache.get(&self.info.site_cache,
                (self.arr_index, self.cache.icentroid+1)
            )
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.icentroid), val.x)
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.icentroid+1), val.y)
        return val

cdef inline Vector2D maxcenter(Site* self, Vector2D val) nogil:
    if isnan(<double>val.x):
        return _Vector2D(
            self.info.site_cache.get(&self.info.site_cache,
                (self.arr_index, self.cache.imaxcenter)
            ),
            self.info.site_cache.get(&self.info.site_cache,
                (self.arr_index, self.cache.imaxcenter+1)
            )
        )
    else:
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.imaxcenter), val.x)
        self.info.site_cache.set(&self.info.site_cache,
            (self.arr_index, self.cache.imaxcenter+1), val.y)
        return val


#### EdgeCacheMap Methods ####

cdef inline EdgeCacheMap _EdgeCacheMap(INT_T iH, INT_T ila, INT_T ida, INT_T iya,
                                       INT_T idVdv, INT_T ila_mag, INT_T ida_mag,
                                       INT_T iya_mag, INT_T icalI, INT_T size) nogil:
    cdef EdgeCacheMap ec
    ec.iH, ec.ila, ec.ida, ec.iya, ec.idVdv = iH, ila, ida, iya, idVdv
    ec.ila_mag, ec.ida_mag, ec.iya_mag, ec.icalI = ila_mag, ida_mag, iya_mag, icalI
    ec.size = size

    ec.H, ec.la, ec.da, ec.ya, ec.dVdv = H, la, da, ya, dVdv
    ec.la_mag, ec.da_mag, ec.ya_mag, ec.calI = la_mag, da_mag, ya_mag, calI

    return ec

cdef inline Matrix2x2 H(HalfEdge* self, Matrix2x2 val) nogil:
    if isnan(<double>val.a):
        return _Matrix2x2(
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.iH)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.iH+1)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.iH+2)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.iH+3)
            ),
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iH), val.a)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iH+1), val.b)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iH+2), val.c)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iH+3), val.d)
        return val

cdef inline Vector2D la(HalfEdge* self, Vector2D val) nogil:
    if isnan(<double>val.x):
        return _Vector2D(
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.ila)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.ila+1)
            )
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.ila), val.x)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.ila+1), val.y)
        return val

cdef inline Vector2D da(HalfEdge* self, Vector2D val) nogil:
    if isnan(<double>val.x):
        return _Vector2D(
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.ida)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.ida+1)
            )
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.ida), val.x)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.ida+1), val.y)
        return val

cdef inline Vector2D ya(HalfEdge* self, Vector2D val) nogil:
    if isnan(<double>val.x):
        return _Vector2D(
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.iya)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.iya+1)
            )
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iya), val.x)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iya+1), val.y)
        return val

cdef inline Vector2D dVdv(HalfEdge* self, Vector2D val) nogil:
    if isnan(<double>val.x):
        return _Vector2D(
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.idVdv)
            ),
            self.info.edge_cache.get(&self.info.edge_cache,
                (self.arr_index, self.cache.idVdv+1)
            )
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.idVdv), val.x)
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.idVdv+1), val.y)
        return val

cdef inline FLOAT_T la_mag(HalfEdge* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.edge_cache.get(&self.info.edge_cache,
            (self.arr_index, self.cache.ila_mag)
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.ila_mag), val)
        return val

cdef inline FLOAT_T da_mag(HalfEdge* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.edge_cache.get(&self.info.edge_cache,
            (self.arr_index, self.cache.ida_mag)
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.ida_mag), val)
        return val

cdef inline FLOAT_T ya_mag(HalfEdge* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.edge_cache.get(&self.info.edge_cache,
            (self.arr_index, self.cache.iya_mag)
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.iya_mag), val)
        return val

cdef inline FLOAT_T calI(HalfEdge* self, FLOAT_T val) nogil:
    if isnan(<double>val):
        return self.info.edge_cache.get(&self.info.edge_cache,
            (self.arr_index, self.cache.icalI)
        )
    else:
        self.info.edge_cache.set(&self.info.edge_cache,
            (self.arr_index, self.cache.icalI), val)
        return val

#### VoronoiInfo Methods ####

cdef inline VoronoiInfo _VoronoiInfo(INT_T [:, ::1] sites, INT_T [:, ::1] edges,
                                     FLOAT_T [:, ::1] points, FLOAT_T [:, ::1] vertices,
                                     FLOAT_T [:, ::1] site_cache,
                                     FLOAT_T [:, ::1] edge_cache,
                                     EdgeCacheMap* edge_cache_map) nogil:
    cdef VoronoiInfo info
    info.sites = _IArray(&sites[0,0], (<INT_T>sites.shape[0], <INT_T>sites.shape[1]))
    info.edges = _IArray(&edges[0,0], (<INT_T>edges.shape[0], <INT_T>edges.shape[1]))
    info.points = _FArray(
        &points[0,0],
        (<INT_T>points.shape[0], <INT_T>points.shape[1])
    )
    info.vertices = _FArray(
        &vertices[0,0],
        (<INT_T>vertices.shape[0], <INT_T>vertices.shape[1])
    )
    info.site_cache = _FArray(
        &site_cache[0,0],
        (<INT_T>site_cache.shape[0], <INT_T>site_cache.shape[1])
    )
    info.edge_cache = _FArray(
        &edge_cache[0,0],
        (<INT_T>edge_cache.shape[0], <INT_T>edge_cache.shape[1])
    )
    info.edge_cache_map = edge_cache_map

    return info


#### Site Methods ####

cdef inline Site _Site(INT_T arr_index, VoronoiInfo* info) nogil:
    cdef Site site
    site.arr_index, site.info, site.cache = arr_index, info, &SITE_CACHE_MAP

    site.index, site.vec, site.edge, site.edge_num = index, vec, edge, edge_num

    return site


cdef inline INT_T index(Site* self) nogil:
    return self.info.sites.get(&self.info.sites, (self.arr_index, 0))

cdef inline Vector2D vec(Site* self) nogil:
    return _Vector2D(
        self.info.points.get(&self.info.points, (self.index(self), 0)),
        self.info.points.get(&self.info.points, (self.index(self), 1))
    )

cdef inline HalfEdge edge(Site* self) nogil:
    return _HalfEdge(
        self.info.sites.get(&self.info.sites, (self.arr_index, 1)), self.info
    )

cdef inline INT_T edge_num(Site* self) nogil:
    return self.info.sites.get(&self.info.sites, (self.arr_index, 2))


#### HalfEdge Methods ####

cdef inline HalfEdge _HalfEdge(INT_T arr_index, VoronoiInfo* info) nogil:
    cdef HalfEdge e
    e.arr_index, e.info, e.cache = arr_index, info, info.edge_cache_map
    e.orig_arr_index = arr_index

    e.origin_index, e.origin, e.face, e.next, e.prev, e.twin, e.get_H = (
        origin_index, origin, face, edge_next, prev, twin, get_H
    )

    return e


cdef inline INT_T origin_index(HalfEdge* self) nogil:
    return self.info.edges.get(&self.info.edges, (self.arr_index, 0))

cdef inline Vector2D origin(HalfEdge* self) nogil:
    return _Vector2D(
        self.info.vertices.get(&self.info.vertices, (self.origin_index(self), 0)),
        self.info.vertices.get(&self.info.vertices, (self.origin_index(self), 1))
    )

cdef inline Site face(HalfEdge* self) nogil:
    return _Site(
        self.info.edges.get(&self.info.edges, (self.arr_index, 1)), self.info
    )

cdef inline HalfEdge edge_next(HalfEdge* self) nogil:

    return _HalfEdge(
        self.info.edges.get(&self.info.edges, (self.arr_index, 2)), self.info
    )

cdef inline HalfEdge prev(HalfEdge* self) nogil:
    return _HalfEdge(
        self.info.edges.get(&self.info.edges, (self.arr_index, 3)), self.info
    )

cdef inline HalfEdge twin(HalfEdge* self) nogil:
    return _HalfEdge(
        self.info.edges.get(&self.info.edges, (self.arr_index, 4)), self.info
    )

cdef inline Matrix2x2 get_H(HalfEdge* self, Site xi) nogil:
    cdef INT_T this_e = self.origin_index(self)
    cdef HalfEdge s_e = xi.edge(&xi)

    for _ in range(xi.edge_num(&xi)):
        if s_e.origin_index(&s_e) == this_e:
            return s_e.cache.H(&s_e, NAN_MATRIX)
        s_e = s_e.next(&s_e)
    return _Matrix2x2(0.0, 0.0, 0.0, 0.0)


cdef class VoronoiContainer:
    """
    Class for Voronoi diagrams, stored in a modified DCEL.
    :param n: [int] how many sites to generate.
    :param w: [float] width of the bounding domain.
    :param h: [float] height of the bounding domain.
    :param r: [float] radius of zero energy circle.
    :param sites: np.ndarray collection of sites.
    """

    def __init__(VoronoiContainer self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
                 object site_arr):
        self.n, self.w, self.h, self.r = n, w, h, r
        self.dim = [w, h]

        self.calculate_voronoi(site_arr.astype(FLOAT))
        self.generate_dcel()

        self.common_cache()
        self.precompute()
        self.calc_grad()
        self.get_statistics()

    cdef void calculate_voronoi(VoronoiContainer self,
                                np.ndarray[FLOAT_T, ndim=2] site_arr) except *:
        """
        Does all necessary computation and caching once points are set.
        :param site_arr: initial points for this container.
        """
        global SYMM
        cdef np.ndarray[FLOAT_T, ndim=2] symm = np.asarray(SYMM).reshape(9,2)
        cdef np.ndarray[FLOAT_T, ndim=1] dim = np.asarray(self.dim)
        cdef np.ndarray[FLOAT_T, ndim=2] full_site_arr = np.empty(
            (self.n*9+8, 2),
            dtype=FLOAT
        )

        # Generate periodic sites and sites that bound periodic sites.
        cdef INT_T i
        for i in range(9):
            full_site_arr[self.n*i:self.n*(i+1)] = site_arr + symm[i]*dim
            if i > 0:
                full_site_arr[9*self.n+i-1] = dim/2 + 2*dim*symm[i]

        # Use SciPy to compute the Voronoi set.
        self.scipy_vor = scipy.spatial.Voronoi(full_site_arr)
        self.points = self.scipy_vor.points
        self.vertices = self.scipy_vor.vertices


    cdef void generate_dcel(VoronoiContainer self) except *:
        cdef array.array int_tmplt = array.array('q', [])

        cdef np.ndarray[INT_T, ndim=1] offsets = np.zeros(self.n*9+1, dtype=INT)
        cdef array.array vert_indices = array.clone(int_tmplt, 0, False)

        # Flatten regions into array, so it can be used later.
        cdef INT_T i
        for i in range(self.n*9):
            verts = self.scipy_vor.regions[self.scipy_vor.point_region[i]]
            offsets[i+1] = offsets[i] + len(verts) # Build offsets.
            vert_indices.extend(array.array('q', verts))    # Flatten

        # Get vertices of original N sites.
        cdef np.ndarray[INT_T, ndim=1] vert_indices_np = np.asarray(vert_indices)
        cdef np.ndarray[INT_T, ndim=1] border_sites = np.unique(np.searchsorted(
            np.asarray(offsets),    # Check indices where below matches would be inserted
            np.nonzero(np.isin(        # Indices of other verts being in bound verts.
                vert_indices_np[offsets[self.n]:],     # Rest of the verts to check.
                np.unique(vert_indices_np[:offsets[self.n]])    # Bound verts
            ))[0] + offsets[self.n],
            side='right'    # If on index == offset_number, should be part of the next site.
        ) - 1)    # Subtract by one to get actual site number.

        cdef INT_T border_num = len(border_sites)

        # Build sites array.
        # [Site Index, Edge Index/Offset, Edge Count]
        self.sites = np.empty((self.n+border_num, 3), dtype=INT)
        self.sites.base[:self.n, 0] = np.arange(self.n, dtype=INT)
        self.sites.base[self.n:, 0] = border_sites
        self.sites.base[:self.n+1, 1] = offsets[:self.n+1]
        for i in range(self.n):
            self.sites[i, 2] = self.sites[i+1, 1] - self.sites[i, 1]

        cdef INT_T edge_count = offsets[self.n]
        cdef INT_T diff
        for i in range(border_num):
            diff = offsets[border_sites[i]+1] - offsets[border_sites[i]]
            edge_count += diff
            self.sites[self.n+i, 2] = diff
            if i < border_num - 1:
                self.sites[self.n+i+1, 1] = self.sites[self.n+i, 1] + diff

        # Build edges array
        # [Origin Index, Site Index, Next Index, Prev Index, Twin Index]
        self.edges = np.empty((edge_count, 5), dtype=INT)
        cdef np.ndarray[INT_T, ndim=1] site_verts
        cdef INT_T j, site_i, edge_i, edge_offset, vert_num, twin_index

        edge_indices = dict()

        for i in range(self.n + border_num):
            site_i = self.sites[i, 0]
            edge_offset = self.sites[i, 1]
            site_verts = vert_indices_np[offsets[site_i]:offsets[site_i+1]]

            # Scipy outputs sorted vertices, but reverse if not counterclockwise.
            if not VoronoiContainer.sign(self.points[site_i],
                    self.vertices[site_verts[0]], self.vertices[site_verts[1]]):
                site_verts = np.flip(site_verts)

            vert_num = offsets[site_i+1] - offsets[site_i]

            for j in range(vert_num):
                edge_i = edge_offset+j
                self.edges[edge_i, 0] = site_verts[j]
                self.edges[edge_i, 1] = i
                # Add vert_num because of C modulo to get always positive.
                self.edges[edge_i, 2] = (j+vert_num+1) % vert_num + edge_offset
                self.edges[edge_i, 3] = (j+vert_num-1) % vert_num + edge_offset

                # Get reversed tuple to theck for twin.
                twin_index = edge_indices.get(
                    (site_verts[(j+1) % vert_num], site_verts[j]
                ), -1)

                self.edges[edge_i, 4] = twin_index
                if twin_index == -1:
                    edge_indices[(site_verts[j], site_verts[(j+1) % vert_num])] = \
                        j + edge_offset
                else:
                    self.edges[twin_index, 4] = j + edge_offset

        self.site_cache = np.empty((self.n + border_num, 7), dtype=FLOAT)
        self.edge_cache = np.empty((edge_count, self.edge_cache_map.size), dtype=FLOAT)


    cdef void common_cache(VoronoiContainer self) except *:
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
            self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        cdef Site xi
        cdef HalfEdge em, ep
        cdef Vector2D p, q, la, da, ya, Rla, centroid, cent_part

        cdef FLOAT_T [:] area = np.zeros(self.sites.shape[0], dtype=FLOAT)
        cdef FLOAT_T [:] perim = np.zeros(self.sites.shape[0], dtype=FLOAT)

        cdef INT_T i, j
        cdef FLOAT_T area_p, la_mag
        for i in prange(self.sites.shape[0], nogil=True):
            xi = _Site(i, &info)
            centroid = _Vector2D(0, 0)
            em = xi.edge(&xi)
            j = 0
            while j < xi.edge_num(&xi):
                ep = em.next(&em)
                p, q = em.origin(&em), ep.origin(&ep)
                # vp - vm, vm - xi
                la, da = q.copy.vsub(&q, p), p.copy.vsub(&p, xi.vec(&xi))
                la_mag = la.mag(&la)
                area_p = la.dot(&la, da.rot(&da))
                Rla = la.rot(&la)
                ya = Rla.copy.smul(&Rla, -2*area_p/la.dot(&la, la))

                # Calculating centroid.
                cent_part = p.copy.vadd(&p, q)
                cent_part.self.vadd(&cent_part, xi.vec(&xi))
                centroid.self.vadd(&centroid, cent_part.copy.smul(&cent_part, area_p))

                # Caching
                em.cache.la(&em, la)
                em.cache.la_mag(&em, la_mag)
                em.cache.da(&em, da)
                em.cache.da_mag(&em, da.mag(&da))
                em.cache.ya(&em, ya)
                em.cache.ya_mag(&em, ya.mag(&ya))

                area[i] += area_p
                perim[i] += la_mag

                em = em.next(&em)
                j = j + 1

            xi.cache.area(&xi, area[i]/2)
            xi.cache.perim(&xi, perim[i])
            xi.cache.isoparam(&xi, 2*PI*area[i]/(perim[i]**2))
            xi.cache.centroid(&xi, centroid.copy.sdiv(&centroid, 3*area[i]))


    @staticmethod
    cdef inline Matrix2x2 calc_H(HalfEdge em, HalfEdge ep) nogil:
        cdef Vector2D xj, xk, Rxjk, v
        cdef Matrix2x2 top

        xj, xk = em.cache.ya(&em, NAN_VECTOR), ep.cache.ya(&ep, NAN_VECTOR)
        Rxjk = xk.copy.vsub(&xk, xj)
        Rxjk.self.smul(&Rxjk, 2)
        Rxjk = Rxjk.rot(&Rxjk)

        v = ep.cache.da(&ep, NAN_VECTOR)
        top = R.copy.smul(&R, xj.dot(&xj, xj) - xk.dot(&xk, xk))
        top.self.msub(&top, _Matrix2x2(v.x*Rxjk.x, v.x*Rxjk.y, v.y*Rxjk.x, v.y*Rxjk.y))
        top.self.sdiv(&top, -Rxjk.dot(&Rxjk, xj))

        return _Matrix2x2(top.a, top.c, top.b, top.d)

    @staticmethod
    cdef inline bint sign(FLOAT_T [::1] ref, FLOAT_T [::1] p, FLOAT_T [::1] q):
        """
        Outputs if p2 - self is counterclockwise of p1 - self.
        :param p1: [List[float]] first vector
        :param p2: [List[float]] second vector
        :return: [bool] returns if counterclockwise.
        """
        return ((q[0] - ref[0])*-(p[1] - ref[1]) + \
                  (q[1] - ref[1])*(p[0] - ref[0])) >= 0

        # global ROT
        # cdef np.ndarray[FLOAT_T, ndim=2] rot = np.asarray(ROT).reshape(2,2)
        # return (q - ref).dot(rot.dot(p - ref)) >= 0

    cdef void precompute(self) except *:
        pass

    cdef void calc_grad(self) except *:
        pass

    cdef void get_statistics(self) except *:
        self.stats = {}
        cache = self.site_cache[:self.n, :]

        self.stats["site_areas"] = np.asarray(cache[:, SITE_CACHE_MAP.iarea])
        self.stats["site_edge_count"] = np.asarray(self.sites[:self.n, 2])

        self.stats["site_isos"] = np.asarray(cache[:, SITE_CACHE_MAP.iisoparam])
        self.stats["site_energies"] = np.asarray(cache[:, SITE_CACHE_MAP.ienergy])
        self.stats["avg_radius"] = np.asarray(cache[:, SITE_CACHE_MAP.iavg_radius])
        self.stats["centroids"] = np.asarray(
            cache[:, SITE_CACHE_MAP.icentroid:SITE_CACHE_MAP.icentroid+2]
        )

        self.stats["isoparam_avg"] = self.stats["site_areas"] / \
                        (PI*self.stats["avg_radius"]**2)

        edges = np.asarray(self.edges)

        mask = np.nonzero(edges[:, 0] != -1)[0]
        all_edges = mask[(mask % 2 == 0)]
        caches = edges[all_edges, 4]

        edge_cache = np.asarray(self.edge_cache)

        self.stats["edge_lengths"] = edge_cache[caches, self.edge_cache_map.ila_mag]

    @property
    def site_arr(self):
        return np.asarray(self.points[:self.n], dtype=FLOAT)

    @property
    def vor_data(self):
        return self.scipy_vor

    @property
    def gradient(self):
        return np.asarray(self.grad, dtype=FLOAT)

    def add_sites(self, add):
        return (self.site_arr + add) % np.asarray(self.dim, dtype=FLOAT)

    def iterate(self, FLOAT_T step):
        k1 = self.gradient
        k2 = self.__class__(self.n, self.w, self.h, self.r,
                self.add_sites(step*k1)
        ).gradient

        return -(step/2)*(k1+k2), -k1


    def approx_hessian(self, d: float) -> np.ndarray:
        """
        Obtains the approximate Hessian.
        :param d: [float] small d for approximation.
        :return: 2Nx2N array that represents Hessian.
        """
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

        return HE

    def radialt_hessian(self) -> np.ndarray:
        HE = np.zeros((2*self.n, 2*self.n))
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
                                             self.vertices, self.site_cache,
                                             self.edge_cache, self.edge_cache_map)
        cdef Site xi, xk
        cdef HalfEdge e, em, ep, fj, fk
        cdef Vector2D temp1, temp2, dau, dapu
        cdef Matrix2x2 sigI, sigJ, sigK, p, q, tempm, toI, toJ, toK

        cdef INT_T i, j, k, z
        for i in range(self.n):
            xi = _Site(i, &info)
            ep = xi.edge(&xi)
            e = ep.prev(&ep)
            j = 0
            while j < xi.edge_num(&xi):
                e.cache.H(&e, VoronoiContainer.calc_H(e, ep))
                e = e.next(&e)
                j = j + 1


        for i in range(self.n):
            xi = _Site(i, &info)
            e = xi.edge(&xi)

            z = 0
            while z < xi.edge_num(&xi):
                em, ep = e.prev(&e), e.next(&e)
                fj, fk = em.twin(&em), e.twin(&e)
                fj, fk = fj.next(&fj), fk.next(&fk)

                xj, xk = fj.face(&fj), fk.face(&fk)
                j, k = xj.index(&xj) % self.n, xk.index(&xk) % self.n
                if k < 0:
                    k = k + self.n
                if j < 0:
                    j = j + self.n

                sigI = e.cache.H(&e, NAN_MATRIX)
                sigJ = fj.cache.H(&fj, NAN_MATRIX)
                sigK = fk.cache.H(&fk, NAN_MATRIX)

                ### Calculating of p
                temp1 = e.cache.la(&e, NAN_VECTOR)
                temp1 = temp1.rot(&temp1)
                temp1.self.sdiv(
                    &temp1,
                    e.cache.la_mag(&e, NAN) * e.cache.ya_mag(&e, NAN) / 2
                )

                dau = e.cache.da(&e, NAN_VECTOR)
                dau.self.sdiv(&dau, e.cache.da_mag(&e, NAN))
                dapu = ep.cache.da(&ep, NAN_VECTOR)
                dapu.self.sdiv(&dapu, ep.cache.da_mag(&ep, NAN))

                temp2 = dapu.copy.vsub(&dapu, dau)
                temp2 = temp2.rot(&temp2)

                p = _Matrix2x2(
                    temp1.x*temp2.x, temp1.x*temp2.y,
                    temp1.y*temp2.x, temp1.y*temp2.y
                )

                ### Calculating of q
                temp2 = e.cache.la(&e, NAN_VECTOR)
                temp1 = temp2.rot(&temp2)
                q = _Matrix2x2(
                    temp1.x*temp2.x, temp1.x*temp2.y,
                    temp1.y*temp2.x, temp1.y*temp2.y
                )
                q.self.sdiv(&q, e.cache.la_mag(&e, NAN)**2)

                q.self.msub(&q, R)
                q.self.smul(
                    &q,
                    e.cache.calI(&e, NAN) / e.cache.la_mag(&e, NAN)
                )

                temp2 = em.cache.la(&em, NAN_VECTOR)
                temp1 = temp2.rot(&temp2)
                tempm = _Matrix2x2(
                    temp1.x*temp2.x, temp1.x*temp2.y,
                    temp1.y*temp2.x, temp1.y*temp2.y
                )
                tempm.self.sdiv(&tempm, em.cache.la_mag(&em, NAN)**2)

                tempm = R.copy.msub(&R, tempm)
                tempm.self.smul(
                    &tempm,
                    em.cache.calI(&em, NAN) / em.cache.la_mag(&em, NAN)
                )

                q.self.madd(&q, tempm)

                # Calculating components that go to the respective sites
                toI = sigI.copy.mmul(&sigI, p.copy.madd(&p, q))
                toI.self.msub(&toI, p)

                toJ = sigJ.copy.mmul(&sigJ, p.copy.madd(&p, q))
                toK = sigK.copy.mmul(&sigK, p.copy.madd(&p, q))

                HE[2*i: 2*(i+1), 2*i: 2*(i+1)] += np.array([[toI.a, toI.b], [toI.c, toI.d]])
                HE[2*i: 2*(i+1), 2*j: 2*(j+1)] += np.array([[toJ.a, toJ.b], [toJ.c, toJ.d]])
                HE[2*i: 2*(i+1), 2*k: 2*(k+1)] += np.array([[toK.a, toK.b], [toK.c, toK.d]])

                e = e.next(&e)
                z = z + 1


        return -2*self.r*HE

    def site_vert_arr(self): # -> List[np.ndarray]
        cdef VoronoiInfo info = _VoronoiInfo(self.sites, self.edges, self.points,
            self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

        cdef INT_T i, j
        cdef Site xi
        cdef HalfEdge e
        cdef Vector2D v

        sites, site_verts = [], []

        for i in range(self.n):
            xi = _Site(i, &info)
            v = xi.vec(&xi)
            sites.append(np.array([v.x, v.y]))
            verts = np.empty((xi.edge_num(&xi), 2))
            e = xi.edge(&xi)
            for j in range(xi.edge_num(&xi)):
                v = e.origin(&e)
                verts[j, 0], verts[j, 1] = v.x, v.y
                e = e.next(&e)

            site_verts.append(verts)

        return sites, site_verts
