cimport numpy as np

# Cython Types.
ctypedef np.int64_t INT_T
ctypedef np.float64_t FLOAT_T

ctypedef (INT_T, INT_T) Pair

# Integer Array psuedo-class for continguous arrays.
cdef struct IArray:
    INT_T* arr
    Pair shape

    INT_T (*get)(IArray*, Pair) nogil
    void (*set)(IArray*, Pair, INT_T) nogil

# Float Array psuedo-class for continguous arrays.
ctypedef struct FArray:
    FLOAT_T* arr
    Pair shape

    FLOAT_T (*get)(FArray*, Pair) nogil
    void (*set)(FArray*, Pair, FLOAT_T) nogil

# Simple append-only dynamic integer array.
# ctypedef struct IList:
#     INT_T* data
#     INT_T size, length

#     void (*append)(IList*, INT_T) nogil
#     void (*free)(IList*) nogil

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
    Vector2D* (*rot)(Vector2D*) nogil

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
    Vector2D (*rot)(Vector2D*) nogil

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
    Matrix2x2* (*T)(Matrix2x2*) nogil

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
    Matrix2x2 (*T)(Matrix2x2*) nogil

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
    Matrix2x2 (*vecmul)(Vector2D*, Vector2D) nogil
    FLOAT_T (*dot)(Vector2D*, Vector2D) nogil
    FLOAT_T (*mag)(Vector2D*) nogil

# Psuedo-class for a 2x2 matrix.
ctypedef struct Matrix2x2:
    FLOAT_T a, b, c, d
    MatrixSelfOps self
    MatrixCopyOps copy

    bint (*equals)(Matrix2x2*, Matrix2x2) nogil
    Vector2D (*vecmul)(Matrix2x2*, Vector2D) nogil

cdef IArray _IArray(INT_T*, Pair) nogil
cdef FArray _FArray(FLOAT_T*, Pair) nogil
cdef BitSet _BitSet(INT_T) nogil
cdef Vector2D _Vector2D(FLOAT_T, FLOAT_T) nogil
cdef Matrix2x2 _Matrix2x2(FLOAT_T, FLOAT_T, FLOAT_T, FLOAT_T) nogil
