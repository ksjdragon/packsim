from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

MODULE_NAME = "packsim"

ext_modules = [
    Extension(
        MODULE_NAME,
        [f'{MODULE_NAME}.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name=MODULE_NAME,
    ext_modules = cythonize(ext_modules, compiler_directives={
        'language_level': 3, 'boundscheck' : False, 'wraparound': False, 'cdivision' : True
    }),
    include_dirs = [numpy.get_include()]
)