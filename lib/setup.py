# setup.py
# Usage: ``python setup.py build_ext --inplace``
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
setup(  name='_tifffile',
        ext_modules=[Extension('_tifffile', ['tifffile.c'],
        include_dirs=[numpy.get_include()],
        cmdclass={'build_ext': build_ext},
        script_args=['build_ext'],
        options={'build_ext': {'inplace': True, 'force': True}},
                             )])