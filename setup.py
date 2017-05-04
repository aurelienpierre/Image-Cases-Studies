from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import os


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    return Extension(
        extName,
        [extPath],
        # your include_dirs must contains the '.' for setup to search all the
        # subfolder of the codeRootFolder
        include_dirs=['.']
    )


extNames = scandir('lib')

extensions = [makeExtension(name) for name in extNames]

setup(
    name="ICS",
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    script_args=['build_ext'],
    options={'build_ext': {'inplace': True, 'force': True}},
    install_requires=["PIL",
                      "numpy",
                      "scipy",
                      "cython",
                      "sympy",
                      "skimage",
                      "warnings",
                      "time",
                      "numba"]
)
