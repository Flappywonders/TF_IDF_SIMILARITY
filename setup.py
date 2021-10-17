from distutils.core import setup
from Cython.Build import cythonize

setup(name='culculate cosine',
      ext_modules=cythonize("cul_cos.pyx"))