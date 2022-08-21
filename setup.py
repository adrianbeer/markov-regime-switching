from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("time_varying_markov_model.pyx")
)