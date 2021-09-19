from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "_squish",
        ["src/_squish.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name="squish",
    ext_modules = cythonize(ext_modules, compiler_directives={
        'language_level': 3, 'boundscheck' : False, 'wraparound': False, 'cdivision' : True
    }),
    include_dirs = [numpy.get_include()]
)