import numpy

from setuptools import setup
from Cython.Build import cythonize

from distutils.extension import Extension


ext_options = {'language_level': "3", "compiler_directives": {"profile": True}, "annotate": True, 'build_dir': 'build'}
exts = [Extension(name='src.CythonFunctions', sources=['./src/CythonFunctions.pyx'], include_dirs=[numpy.get_include()])]

setup(
    name='Egypt ABM',
    version='0.8',
    license='MIT',
    author='Brandon Gower Winter',
    author_email='GWRBRA001@myuct.ac.za',
    description='Egypt Model',
    ext_modules=cythonize(exts, **ext_options),
    zip_safe=False
)
