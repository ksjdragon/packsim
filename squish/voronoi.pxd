cimport numpy as np
from squish.core cimport INT_T, FLOAT_T, IArray, FArray, Vector2D, Matrix2x2

# Psuedo-class that handles caching for sites.
ctypedef struct SiteCacheMap:
    INT_T iarea, iperim, iisoparam, ienergy, iavg_radius, icentroid, imaxcenter

    FLOAT_T (*area)(Site*, FLOAT_T) nogil
    FLOAT_T (*perim)(Site*, FLOAT_T) nogil
    FLOAT_T (*isoparam)(Site*, FLOAT_T) nogil
    FLOAT_T (*energy)(Site*, FLOAT_T) nogil
    FLOAT_T (*avg_radius)(Site*, FLOAT_T) nogil

    Vector2D (*centroid)(Site*, Vector2D) nogil
    Vector2D (*maxcenter)(Site*, Vector2D) nogil

# Psuedo-class that handles caching for edges.
ctypedef struct EdgeCacheMap:
    INT_T iH, ila, ida, iya, idVdv, ila_mag, ida_mag, iya_mag, icalI, size

    Matrix2x2 (*H)(HalfEdge*, Matrix2x2) nogil

    Vector2D (*la)(HalfEdge*, Vector2D) nogil
    Vector2D (*da)(HalfEdge*, Vector2D) nogil
    Vector2D (*ya)(HalfEdge*, Vector2D) nogil
    Vector2D (*dVdv)(HalfEdge*, Vector2D) nogil

    FLOAT_T (*la_mag)(HalfEdge*, FLOAT_T) nogil
    FLOAT_T (*da_mag)(HalfEdge*, FLOAT_T) nogil
    FLOAT_T (*ya_mag)(HalfEdge*, FLOAT_T) nogil
    FLOAT_T (*calI)(HalfEdge*, FLOAT_T) nogil

# Psuedo-class to just contain all pertaining info for sites and edges.
ctypedef struct VoronoiInfo:
    IArray sites, edges
    FArray points, vertices, site_cache, edge_cache
    EdgeCacheMap* edge_cache_map

# Psuedo-class for a Site.
ctypedef struct Site:
    INT_T arr_index
    VoronoiInfo* info
    SiteCacheMap* cache

    INT_T (*index)(Site*) nogil
    Vector2D (*vec)(Site*) nogil
    HalfEdge (*edge)(Site*) nogil
    INT_T (*edge_num)(Site*) nogil

# Psuedo-class for an HalfEdge.
ctypedef struct HalfEdge:
    INT_T orig_arr_index, arr_index
    VoronoiInfo* info
    EdgeCacheMap* cache

    INT_T (*origin_index)(HalfEdge*) nogil
    Vector2D (*origin)(HalfEdge*) nogil
    Site (*face)(HalfEdge*) nogil
    HalfEdge (*next)(HalfEdge*) nogil
    HalfEdge (*prev)(HalfEdge*) nogil
    HalfEdge (*twin)(HalfEdge*) nogil
    Matrix2x2 (*get_H)(HalfEdge*, Site) nogil


cdef class VoronoiContainer:
    cdef readonly INT_T n
    cdef readonly FLOAT_T w, h, r, energy
    cdef FLOAT_T [2] dim
    cdef FLOAT_T [:, ::1] points, vertices, site_cache, edge_cache, grad
    cdef INT_T [:, ::1] sites, edges
    cdef EdgeCacheMap* edge_cache_map
    cdef dict __dict__

    cdef void calculate_voronoi(VoronoiContainer self,
                                np.ndarray[FLOAT_T, ndim=2] site_arr) except *
    cdef void generate_dcel(VoronoiContainer self) except *
    cdef void common_cache(VoronoiContainer self) except *
    cdef void precompute(self) except *
    cdef void calc_grad(self) except *
    cdef void get_statistics(VoronoiContainer self) except *

    @staticmethod
    cdef inline Matrix2x2 calc_H(HalfEdge, HalfEdge) nogil
    @staticmethod
    cdef inline bint sign(FLOAT_T [::1], FLOAT_T [::1], FLOAT_T [::1])



cdef SiteCacheMap _SiteCacheMap(INT_T, INT_T, INT_T, INT_T, INT_T, INT_T, INT_T) nogil
cdef EdgeCacheMap _EdgeCacheMap(INT_T, INT_T, INT_T, INT_T, INT_T, INT_T, INT_T,
                                INT_T, INT_T, INT_T) nogil
cdef VoronoiInfo _VoronoiInfo(INT_T [:, ::1], INT_T[:, ::1], FLOAT_T[:, ::1],
                              FLOAT_T[:, ::1], FLOAT_T[:, ::1], FLOAT_T[:, ::1],
                              EdgeCacheMap*) nogil
cdef Site _Site(INT_T, VoronoiInfo*) nogil
cdef HalfEdge _HalfEdge(INT_T, VoronoiInfo*) nogil

cdef Vector2D NAN_VECTOR
cdef Matrix2x2 R, NAN_MATRIX
