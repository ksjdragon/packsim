from setuptools import Extension, setup
import numpy

try:
	from Cython.Build import cythonize
	USE_CYTHON = True
except ImportError:
	USE_CYTHON = False

if USE_CYTHON:
	ext_modules = cythonize([
		Extension(
			"_squish",
			["squish/_squish/_squish.pyx"],
			extra_compile_args=['-fopenmp'],
			extra_link_args=['-fopenmp']
		)
	],
	compiler_directives={
		'language_level': 3, 'boundscheck' : False, 'wraparound': False, 'cdivision' : True
	})
else:
	ext_modules = [
	Extension('squish._squish', ["squish/_squish/_squish.c"])
]

#annotate='fullc'
setup(
	ext_modules = ext_modules,
	include_dirs = [numpy.get_include()]
)