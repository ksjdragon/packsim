cimport numpy as np

# Cython Types.
ctypedef np.int64_t INT_T
ctypedef np.float64_t FLOAT_T

# Stores initialization functions.
cdef struct Init:
	IArray (*IArray)(INT_T*, (INT_T, INT_T)) nogil
	FArray (*FArray)(FLOAT_T*, (INT_T, INT_T)) nogil
	#IList (*IList)() nogil
	BitSet (*BitSet)(INT_T) nogil
	Vector2D (*Vector2D)(FLOAT_T, FLOAT_T) nogil
	Matrix2x2 (*Matrix2x2)(FLOAT_T, FLOAT_T, FLOAT_T, FLOAT_T) nogil
	SiteCacheMap (*SiteCacheMap)(INT_T, INT_T, INT_T, INT_T, INT_T) nogil
	EdgeCacheMap (*EdgeCacheMap)(INT_T, INT_T, INT_T, INT_T, INT_T, INT_T, INT_T, INT_T,
					INT_T, INT_T, INT_T, INT_T, INT_T, INT_T) nogil
	VoronoiInfo (*VoronoiInfo)(INT_T [:, ::1], INT_T[:, ::1], FLOAT_T[:, ::1], 
								FLOAT_T[:, ::1], FLOAT_T[:, ::1], FLOAT_T[:, ::1],
								EdgeCacheMap*) nogil
	Site (*Site)(INT_T, VoronoiInfo*) nogil
	HalfEdge (*HalfEdge)(INT_T, VoronoiInfo*) nogil

# Integer Array psuedo-class for continguous arrays.
cdef struct IArray:
	INT_T* arr
	(INT_T, INT_T) shape

	INT_T (*get)(IArray*, (INT_T, INT_T)) nogil
	void (*set)(IArray*, (INT_T, INT_T), INT_T) nogil

# Float Array psuedo-class for continguous arrays.
ctypedef struct FArray:
	FLOAT_T* arr
	(INT_T, INT_T) shape

	FLOAT_T (*get)(FArray*, (INT_T, INT_T)) nogil
	void (*set)(FArray*, (INT_T, INT_T), FLOAT_T) nogil

# Simple append-only dynamic integer array.
# ctypedef struct IList:
# 	INT_T* data
# 	INT_T size, length

# 	void (*append)(IList*, INT_T) nogil
# 	void (*free)(IList*) nogil

# Uses an array of bits to determine if value in set.
ctypedef struct BitSet:
	INT_T* bits

	bint (*add)(BitSet*, INT_T) nogil
	void (*free)(BitSet*) nogil

# Psuedo-operator definitions.
ctypedef Vector2D* (*VectorSelfVecOp)(Vector2D*, Vector2D) nogil
ctypedef Vector2D (*VectorCopyVecOp)(Vector2D*, Vector2D) nogil
ctypedef Vector2D* (*VectorSelfSclOp)(Vector2D*, FLOAT_T) nogil
ctypedef Vector2D (*VectorCopySclOp)(Vector2D*, FLOAT_T) nogil

ctypedef Matrix2x2* (*MatrixSelfMatOp)(Matrix2x2*, Matrix2x2) nogil
ctypedef Matrix2x2 (*MatrixCopyMatOp)(Matrix2x2*, Matrix2x2) nogil
ctypedef Matrix2x2* (*MatrixSelfSclOp)(Matrix2x2*, FLOAT_T) nogil
ctypedef Matrix2x2 (*MatrixCopySclOp)(Matrix2x2*, FLOAT_T) nogil


ctypedef struct VectorSelfOps:
	Vector2D* (*neg)(Vector2D*) nogil

	VectorSelfVecOp vadd
	VectorSelfVecOp vsub
	VectorSelfVecOp vmul
	VectorSelfVecOp vdiv
	Vector2D* (*matmul)(Vector2D*, Matrix2x2) nogil

	VectorSelfSclOp sadd
	VectorSelfSclOp ssub
	VectorSelfSclOp smul
	VectorSelfSclOp sdiv


ctypedef struct VectorCopyOps:
	Vector2D (*neg)(Vector2D*) nogil

	VectorCopyVecOp vadd
	VectorCopyVecOp vsub
	VectorCopyVecOp vmul
	VectorCopyVecOp vdiv
	Vector2D (*matmul)(Vector2D*, Matrix2x2) nogil

	VectorCopySclOp sadd
	VectorCopySclOp ssub
	VectorCopySclOp smul
	VectorCopySclOp sdiv


ctypedef struct MatrixSelfOps:
	Matrix2x2* (*neg)(Matrix2x2*) nogil

	MatrixSelfMatOp madd
	MatrixSelfMatOp msub
	MatrixSelfMatOp mmul
	MatrixSelfMatOp mdiv
	MatrixSelfMatOp matmul

	MatrixSelfSclOp sadd
	MatrixSelfSclOp ssub
	MatrixSelfSclOp smul
	MatrixSelfSclOp sdiv


ctypedef struct MatrixCopyOps:
	Matrix2x2 (*neg)(Matrix2x2*) nogil

	MatrixCopyMatOp madd
	MatrixCopyMatOp msub
	MatrixCopyMatOp mmul
	MatrixCopyMatOp mdiv
	MatrixCopyMatOp matmul

	MatrixCopySclOp sadd
	MatrixCopySclOp ssub
	MatrixCopySclOp smul
	MatrixCopySclOp sdiv

# Psuedo-class for a 2-dimensional vector. No orientation.
ctypedef struct Vector2D:
	FLOAT_T x, y
	VectorSelfOps self
	VectorCopyOps copy

	bint (*equals)(Vector2D*, Vector2D) nogil
	Vector2D (*rot)(Vector2D*) nogil
	FLOAT_T (*dot)(Vector2D*, Vector2D) nogil
	FLOAT_T (*mag)(Vector2D*) nogil

# Psuedo-class for a 2x2 matrix.
ctypedef struct Matrix2x2:
	FLOAT_T a, b, c, d
	MatrixSelfOps self
	MatrixCopyOps copy

	bint (*equals)(Matrix2x2*, Matrix2x2) nogil
	Vector2D (*vecmul)(Matrix2x2*, Vector2D) nogil

# Psuedo-class that handles caching for sites.
ctypedef struct SiteCacheMap:
	INT_T iarea, iperim, iisoparam, ienergy, iavg_radius

	FLOAT_T (*area)(Site*, FLOAT_T) nogil
	FLOAT_T (*perim)(Site*, FLOAT_T) nogil
	FLOAT_T (*isoparam)(Site*, FLOAT_T) nogil
	FLOAT_T (*energy)(Site*, FLOAT_T) nogil
	FLOAT_T (*avg_radius)(Site*, FLOAT_T) nogil

# Psuedo-class that handles caching for edges.
ctypedef struct EdgeCacheMap:
	INT_T iH, ila, ila_mag, ida, ida_mag, ixij, idVdv, iphi, iB, iF, ii2p,\
			ilntan, icsc, size

	Matrix2x2 (*H)(HalfEdge*, Matrix2x2) nogil

	Vector2D (*la)(HalfEdge*, Vector2D) nogil
	Vector2D (*da)(HalfEdge*, Vector2D) nogil
	Vector2D (*xij)(HalfEdge*, Vector2D) nogil
	Vector2D (*dVdv)(HalfEdge*, Vector2D) nogil
	Vector2D (*i2p)(HalfEdge*, Vector2D) nogil

	FLOAT_T (*la_mag)(HalfEdge*, FLOAT_T) nogil
	FLOAT_T (*da_mag)(HalfEdge*, FLOAT_T) nogil
	FLOAT_T (*phi)(HalfEdge*, FLOAT_T) nogil
	FLOAT_T (*B)(HalfEdge*, FLOAT_T) nogil
	FLOAT_T (*F)(HalfEdge*, FLOAT_T) nogil	
	FLOAT_T (*lntan)(HalfEdge*, FLOAT_T) nogil
	FLOAT_T (*csc)(HalfEdge*, FLOAT_T) nogil

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


cdef class AreaEnergy(VoronoiContainer):
	cdef readonly FLOAT_T minimum
	cdef void precompute(self) except *
	cdef void calc_grad(self) except *


cdef class RadialALEnergy(VoronoiContainer):
	cdef void precompute(self) except *
	cdef void calc_grad(self) except *


cdef class RadialTEnergy(VoronoiContainer):
	cdef void precompute(self) except *
	cdef void calc_grad(self) except *

cdef class Calc:
	@staticmethod
	cdef inline FLOAT_T phi(HalfEdge) nogil
	@staticmethod
	cdef inline Vector2D I2(HalfEdge, FLOAT_T, FLOAT_T) nogil
	@staticmethod
	cdef Vector2D radialt_edge_grad(HalfEdge, Site, FLOAT_T) nogil