import array, scipy.spatial, numpy as np
from cython.parallel import parallel, prange

cimport numpy as np
from cpython cimport array
from libc.stdlib cimport malloc, realloc, calloc, free
from libc.math cimport isnan, NAN, pi as PI, M_PI_2 as PI_2, \
				sqrt, log, sin, cos, tan, acos, fabs
from packsim_core cimport INT_T, FLOAT_T, Init, IArray, FArray, BitSet, Vector2D, Matrix2x2, \
					VectorSelfOps, VectorCopyOps, MatrixSelfOps, MatrixCopyOps, \
					SiteCacheMap, EdgeCacheMap, VoronoiInfo, Site, HalfEdge

#### Constants ####

INT = np.int64
FLOAT = np.float64

cdef FLOAT_T TAU = 2*PI
# In most cases, the amount of edges relevant to a gradient will
# not exceed this number. However, we assign a growth rate of 8 edges,
# when dynamically allocating. 
cdef INT_T EDGE_ARR_SIZE = 32 

cdef Init init
init.IArray, init.FArray, init.BitSet, init.Vector2D, init.Matrix2x2 = \
	init_iarray, init_farray, init_bitset, init_vector2d, init_matrix2x2

cdef VectorSelfOps VSO
cdef VectorCopyOps VCO
cdef MatrixSelfOps MSO
cdef MatrixCopyOps MCO

VSO.neg, VSO.vadd, VSO.vsub, VSO.vmul, VSO.vdiv, VSO.sadd, VSO.ssub, VSO.smul, VSO.sdiv = \
	v_neg_s, v_vadd_s, v_vsub_s, v_vmul_s, v_vdiv_s, v_sadd_s, v_ssub_s, v_smul_s, v_sdiv_s
VSO.matmul = v_matmul_s

VCO.neg, VCO.vadd, VCO.vsub, VCO.vmul, VCO.vdiv, VCO.sadd, VCO.ssub, VCO.smul, VCO.sdiv = \
	v_neg_c, v_vadd_c, v_vsub_c, v_vmul_c, v_vdiv_c, v_sadd_c, v_ssub_c, v_smul_c, v_sdiv_c
VCO.matmul = v_matmul_c

MSO.neg, MSO.madd, MSO.msub, MSO.mmul, MSO.mdiv, MSO.sadd, MSO.ssub, MSO.smul, MSO.sdiv = \
	m_neg_s, m_madd_s, m_msub_s, m_mmul_s, m_mdiv_s, m_sadd_s, m_ssub_s, m_smul_s, m_sdiv_s
MSO.matmul = m_matmul_s

MCO.neg, MCO.madd, MCO.msub, MCO.mmul, MCO.mdiv, MCO.sadd, MCO.ssub, MCO.smul, MCO.sdiv = \
	m_neg_c, m_madd_c, m_msub_c, m_mmul_c, m_mdiv_c, m_sadd_c, m_ssub_c, m_smul_c, m_sdiv_c
MCO.matmul = m_matmul_c

cdef Vector2D NAN_VECTOR = init.Vector2D(NAN, NAN)
cdef Matrix2x2 NAN_MATRIX = init.Matrix2x2(NAN, NAN, NAN, NAN)

cdef FLOAT_T[18] SYMM = [0,0, 1,0, 1,1, 0,1, -1,1, -1,0, -1,-1, 0,-1, 1,-1]
cdef Matrix2x2 R = init.Matrix2x2(0, -1, 1, 0)

"""
If bound checking is desired, uncomment out ..._valid_indices functions.
"""

#### IArray Methods ####

cdef inline IArray init_iarray(INT_T* arr, (INT_T, INT_T) shape) nogil:
	cdef IArray iarray
	iarray.arr, iarray.shape = arr, shape

	iarray.get = iarray_get
	iarray.set = iarray_set
	return iarray

cdef inline bint iarray_valid_indices(IArray* self, (INT_T, INT_T) index) nogil:
	if index[0] > self.shape[0] or index[1] > self.shape[1]:
		with gil:
			raise IndexError(f"Index out of range for IArray with shape {self.shape}")

cdef inline INT_T iarray_get(IArray* self, (INT_T, INT_T) index) nogil:
	#iarray_valid_indices(&self, index)
	return self.arr[index[0]*self.shape[1] + index[1]]

cdef inline void iarray_set(IArray* self, (INT_T, INT_T) index, INT_T val) nogil:
	#iarray_valid_indices(&self, index)
	self.arr[index[0]*self.shape[1] + index[1]] = val 


#### FArray Methods ####

cdef inline FArray init_farray(FLOAT_T* arr, (INT_T, INT_T) shape) nogil:
	cdef FArray farray
	farray.arr, farray.shape = arr, shape

	farray.get = farray_get
	farray.set = farray_set
	return farray

cdef inline bint farray_valid_indices(FArray* self, (INT_T, INT_T) index) nogil:
	if index[0] > self.shape[0] or index[1] > self.shape[1]:
		with gil:
			raise IndexError(f"Index out of range for FArray with shape {self.shape}")

cdef inline FLOAT_T farray_get(FArray* self, (INT_T, INT_T) index) nogil:
	#iarray_valid_indices(&self, index)
	return self.arr[index[0]*self.shape[1] + index[1]]

cdef inline void farray_set(FArray* self, (INT_T, INT_T) index, FLOAT_T val) nogil:
	#iarray_valid_indices(&self, index)
	self.arr[index[0]*self.shape[1] + index[1]] = val


#### IList Methods ####

# cdef inline IList init_ilist() nogil:
# 	cdef IList ilist
# 	ilist.size = EDGE_ARR_SIZE
# 	ilist.length = 0
# 	ilist.data = <INT_T*> malloc(self.size * sizeof(INT_T))

# 	ilist.append, ilist.free = ilist_append, ilist_free

# 	return ilist

# cdef inline void ilist_append(IList* self, INT_T) nogil:
# 	if self.size == self.length:
# 		ilist.data = <INT_T*> realloc((self.size+8) * sizeof(INT_T))
# 		self.size += 8

# 	self.data[self.length] == INT_T
# 	self.length += 1

# cdef inline void ilist_free(IList* self) nogil:
# 	free(self.data)

#### BitSet Methods ####

cdef inline BitSet init_bitset(INT_T elements) nogil:
	cdef BitSet bitset
	bitset.bits = <INT_T*> calloc(((elements/sizeof(INT_T))+1), sizeof(INT_T))

	bitset.add, bitset.free = bitset_add, bitset_free
	return bitset

cdef inline bint bitset_add(BitSet* self, INT_T val) nogil:
	cdef INT_T index, rel_index, old
	index = val/sizeof(INT_T)
	old = self.bits[index]
	rel_index = val - index*sizeof(INT_T)

	self.bits[index] = (1 << rel_index) | old	# New value.

	return old == self.bits[index]	# Means 1 was already there.

cdef inline void bitset_free(BitSet* self) nogil:
	free(self.bits)

#### Vector2D Methods ####
"""
Prefix 'v' stands for vector, element by element operation.
Prefix 's' stands for scalar, broadcasted operation.
Suffix 'w' stands for write, overwriting current value.
Suffix 'n' stands for new, copying to a new location.

While it's possible to chain 'new' operations, when possible, 
avoid this, so fewer objects are needed.
"""

cdef inline Vector2D init_vector2d(FLOAT_T x, FLOAT_T y) nogil:
	cdef Vector2D vec
	vec.x, vec.y = x, y
	vec.self, vec.copy = VSO, VCO
	
	vec.equals, vec.rot, vec.dot, vec.mag = v_equals, rot, dot, mag
	
	return vec


cdef inline bint v_equals(Vector2D* self, Vector2D w) nogil:
	return ((self.x == w.x) and (self.y == w.y))

cdef inline Vector2D* v_neg_s(Vector2D* self) nogil:
	self.x = -self.x
	self.y = -self.y
	return self

cdef inline Vector2D* v_vadd_s(Vector2D* self, Vector2D w) nogil:
	self.x += w.x
	self.y += w.y
	return self

cdef inline Vector2D* v_vsub_s(Vector2D* self, Vector2D w) nogil:
	self.x -= w.x
	self.y -= w.y
	return self

cdef inline Vector2D* v_vmul_s(Vector2D* self, Vector2D w) nogil:
	self.x *= w.x
	self.y *= w.y
	return self

cdef inline Vector2D* v_vdiv_s(Vector2D* self, Vector2D w) nogil:
	self.x /= w.x
	self.y /= w.y
	return self

cdef inline Vector2D* v_sadd_s(Vector2D* self, FLOAT_T s) nogil:
	self.x += s
	self.y += s
	return self

cdef inline Vector2D* v_ssub_s(Vector2D* self, FLOAT_T s) nogil:
	self.x -= s
	self.y -= s
	return self

cdef inline Vector2D* v_smul_s(Vector2D* self, FLOAT_T s) nogil:
	self.x *= s
	self.y *= s
	return self

cdef inline Vector2D* v_sdiv_s(Vector2D* self, FLOAT_T s) nogil:
	self.x /= s
	self.y /= s
	return self

cdef inline Vector2D* v_matmul_s(Vector2D* self, Matrix2x2 m) nogil:
	self.x, self.y = self.x*m.a + self.y*m.c, self.x*m.b + self.y*m.d
	return self 

cdef inline Vector2D v_neg_c(Vector2D* self) nogil:
	return init.Vector2D(-self.x, -self.y)

cdef inline Vector2D v_vadd_c(Vector2D* self, Vector2D w) nogil:
	return init.Vector2D(self.x + w.x, self.y + w.y)

cdef inline Vector2D v_vsub_c(Vector2D* self, Vector2D w) nogil:
	return init.Vector2D(self.x - w.x, self.y - w.y)

cdef inline Vector2D v_vmul_c(Vector2D* self, Vector2D w) nogil:
	return init.Vector2D(self.x * w.x, self.y * w.y)

cdef inline Vector2D v_vdiv_c(Vector2D* self, Vector2D w) nogil:
	return init.Vector2D(self.x / w.x, self.y / w.y)

cdef inline Vector2D v_sadd_c(Vector2D* self, FLOAT_T s) nogil:
	return init.Vector2D(self.x + s, self.y + s)

cdef inline Vector2D v_ssub_c(Vector2D* self, FLOAT_T s) nogil:
	return init.Vector2D(self.x + s, self.y + s)

cdef inline Vector2D v_smul_c(Vector2D* self, FLOAT_T s) nogil:
	return init.Vector2D(self.x * s, self.y * s)

cdef inline Vector2D v_sdiv_c(Vector2D* self, FLOAT_T s) nogil:
	return init.Vector2D(self.x / s, self.y / s)

cdef inline Vector2D v_matmul_c(Vector2D* self, Matrix2x2 m) nogil:
	return init.Vector2D(
		self.x*m.a + self.y*m.c, self.x*m.b + self.y*m.d 
	)

cdef inline Vector2D rot(Vector2D* self) nogil:
	return init.Vector2D(-self.y, self.x)

cdef inline FLOAT_T dot(Vector2D* self, Vector2D w) nogil:
	return self.x*w.x + self.y*w.y

cdef inline FLOAT_T mag(Vector2D* self) nogil:
	return <FLOAT_T>sqrt(<double>(self.x*self.x + self.y*self.y))


#### Matrix2x2 Methods ####

cdef inline Matrix2x2 init_matrix2x2(FLOAT_T a, FLOAT_T b, FLOAT_T c, FLOAT_T d) nogil:
	cdef Matrix2x2 matrix
	matrix.a, matrix.b, matrix.c, matrix.d = a, b, c, d
	matrix.self, matrix.copy = MSO, MCO
	
	matrix.equals, matrix.vecmul = m_equals, m_vecmul

	return matrix

cdef inline bint m_equals(Matrix2x2* self, Matrix2x2 m) nogil:
	return (
		(self.a == m.a) and (self.b == m.b) and (self.c == m.c) and (self.d == m.d)
	)

cdef inline Vector2D m_vecmul(Matrix2x2* self, Vector2D v) nogil:
	return init.Vector2D(
		self.a*v.x + self.b*v.y, self.c*v.x + self.d*v.y
	)

cdef inline Matrix2x2* m_neg_s(Matrix2x2* self) nogil:
	self.a, self.b, self.c, self.d = -self.a, -self.b, -self.c, -self.d
	return self 

cdef inline Matrix2x2* m_madd_s(Matrix2x2* self, Matrix2x2 m) nogil:
	self.a += m.a
	self.b += m.b
	self.c += m.c
	self.d += m.d
	return self

cdef inline Matrix2x2* m_msub_s(Matrix2x2* self, Matrix2x2 m) nogil:
	self.a -= m.a
	self.b -= m.b
	self.c -= m.c
	self.d -= m.d
	return self

cdef inline Matrix2x2* m_mmul_s(Matrix2x2* self, Matrix2x2 m) nogil:
	self.a *= m.a
	self.b *= m.b
	self.c *= m.c
	self.d *= m.d
	return self

cdef inline Matrix2x2* m_mdiv_s(Matrix2x2* self, Matrix2x2 m) nogil:
	self.a /= m.a
	self.b /= m.b
	self.c /= m.c
	self.d /= m.d
	return self

cdef inline Matrix2x2* m_sadd_s(Matrix2x2* self, FLOAT_T s) nogil:
	self.a += s
	self.b += s
	self.c += s
	self.d += s
	return self

cdef inline Matrix2x2* m_ssub_s(Matrix2x2* self, FLOAT_T s) nogil:
	self.a -= s
	self.b -= s
	self.c -= s
	self.d -= s
	return self

cdef inline Matrix2x2* m_smul_s(Matrix2x2* self, FLOAT_T s) nogil:
	self.a *= s
	self.b *= s
	self.c *= s
	self.d *= s
	return self

cdef inline Matrix2x2* m_sdiv_s(Matrix2x2* self, FLOAT_T s) nogil:
	self.a /= s
	self.b /= s
	self.c /= s
	self.d /= s
	return self

cdef inline Matrix2x2* m_matmul_s(Matrix2x2* self, Matrix2x2 m) nogil:
	self.a, self.b, self.c, self.d = \
		self.a*m.a + self.b*m.c, self.a*m.b + self.b*m.d, \
		self.c*m.a + self.d*m.c, self.c*m.b + self.d*m.d
	return self

cdef inline Matrix2x2 m_neg_c(Matrix2x2* self) nogil:
	return init.Matrix2x2(-self.a, -self.b, -self.c, -self.d)

cdef inline Matrix2x2 m_madd_c(Matrix2x2* self, Matrix2x2 m) nogil:
	return init.Matrix2x2(self.a+m.a, self.b+m.b, self.c+m.c, self.d+m.d) 

cdef inline Matrix2x2 m_msub_c(Matrix2x2* self, Matrix2x2 m) nogil:
	return init.Matrix2x2(self.a-m.a, self.b-m.b, self.c-m.c, self.d-m.d)

cdef inline Matrix2x2 m_mmul_c(Matrix2x2* self, Matrix2x2 m) nogil:
	return init.Matrix2x2(self.a*m.a, self.b*m.b, self.c*m.c, self.d*m.d) 

cdef inline Matrix2x2 m_mdiv_c(Matrix2x2* self, Matrix2x2 m) nogil:
	return init.Matrix2x2(self.a/m.a, self.b/m.b, self.c/m.c, self.d/m.d)

cdef inline Matrix2x2 m_sadd_c(Matrix2x2* self, FLOAT_T s) nogil:
	return init.Matrix2x2(self.a+s, self.b+s, self.c+s, self.d+s) 

cdef inline Matrix2x2 m_ssub_c(Matrix2x2* self, FLOAT_T s) nogil:
	return init.Matrix2x2(self.a-s, self.b-s, self.c-s, self.d-s)

cdef inline Matrix2x2 m_smul_c(Matrix2x2* self, FLOAT_T s) nogil:
	return init.Matrix2x2(self.a*s, self.b*s, self.c*s, self.d*s)

cdef inline Matrix2x2 m_sdiv_c(Matrix2x2* self, FLOAT_T s) nogil:
	return init.Matrix2x2(self.a/s, self.b/s, self.c/s, self.d/s)

cdef inline Matrix2x2 m_matmul_c(Matrix2x2* self, Matrix2x2 m) nogil:
	return init.Matrix2x2(
		self.a*m.a + self.b*m.c, self.a*m.b + self.b*m.d,
		self.c*m.a + self.d*m.c, self.c*m.b + self.d*m.d 
	)