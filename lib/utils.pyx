'''
Created on 27 avr. 2017

@author: aurelien

Source : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3166524/
'''
from PIL import Image
from scipy import interpolate
from skimage import color
from sympy import *
from sympy.matrices import *
from threading import Thread
import os
import scipy.signal
import time
import warnings

import numpy as np


cimport numpy as np
from numpy cimport ndarray


DTYPE = np.float
ctypedef np.float_t DTYPE_t


def timeit(method):
    '''
    From: http://www.samuelbosch.com/2012/02/timing-functions-in-python.html
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed


cdef class colors:
    """
    Stores images as LAB and RGB channels and synchronize them
    """
    cdef public np.ndarray RGB, LAB

    cdef np.ndarray compute_LAB(self):
        self.LAB = color.rgb2lab(self.RGB / 255.0)

    cdef np.ndarray compute_RGB(self):
        self.RGB = (color.lab2rgb(self.LAB) * 255).astype(np.uint8)

    @property
    def L(self):
        return self.LAB[..., 0]

    @property
    def A(self):
        return self.LAB[..., 1]

    @property
    def B(self):
        return self.LAB[..., 2]

    @L.setter
    def L(self, L):
        self.LAB[..., 0] = L
        self.compute_RGB()

    @A.setter
    def A(self, A):
        self.LAB[..., 1] = A
        self.compute_RGB()

    @B.setter
    def B(self, B):
        self.LAB[..., 2] = B
        self.compute_RGB()

    def __init__(self, picture):
        self.RGB = np.array(picture)
        self.compute_LAB()


@timeit
def image_open(src):
    """
    Expose Cython colors class to Python
    """
    return colors(src)


cdef np.ndarray Lagrange_interpolation(np.ndarray points, str variable=None):
    """
    Compute the Lagrange interpolation polynomial.

    :var points: A numpy n×2 ndarray of the interpolations points
    :var variable: None, float or ndarray
    :returns:   * P the symbolic expression
                * Y the evaluation of the polynomial if `variable` is float or ndarray

    """

    x = Symbol("x")
    L = zeros(1, points.shape[0])
    cdef int i = 0

    for p in points:
        numerator = 1
        denominator = 1
        other_points = np.delete(points, i, 0)

        for other_p in other_points:
            numerator = numerator * (x - other_p[0])
            denominator = denominator * (p[0] - other_p[0])

        L[i] = numerator / denominator
        i = i + 1

    P = horner(L.multiply(points[..., 1])[0])
    Y = None

    try:
        Y = lambdify(x, P, 'numpy')
        Y = Y(variable)

    except:
        warnings.warn(
            "No input variable given - polynomial evaluation skipped")

    return P, Y


def histogram(src):
    """
    Fit the histogram entirely
    """
    delta = 100 / (np.max(src.L) - np.min(src.L))
    src.L = (src.L - np.min(src.L)) * delta
    src.A = src.A * delta
    src.B = src.B * delta
    return src


def grey_point(colors src, float amount):
    """
    Adjust the grey point to the desired amount using Lagrange interpolation

    """

    cdef np.ndarray set1 = np.array([  # Quadratic
        [0, 1],
        [amount, amount],
        [100, 100]
    ])

    cdef np.ndarray set2 = np.array([  # Quadratic
        [0, 1],
        [src.L.mean(), amount],
        [100, 100]
    ])

    print("Original grey point : %i %%" % src.L.mean())

    P1, Y1 = Lagrange_interpolation(set1, src.L)
    P2, Y2 = Lagrange_interpolation(set2, src.L)

    src.L = src.L * Y2 / Y1
    src.A = src.A * Y2 / Y1
    src.B = src.B * Y2 / Y1

    print("Actual grey point : %i %%" % src.L.mean())
    return src


def auto_vibrance(colors src):
    """
    Add some more saturation preserving the skin tones (A < 20, B < 18)
    """
    cdef np.ndarray x1 = np.array([-100, -50, -20, 0, 20, 50, 100])
    cdef np.ndarray y1 = np.array([100,  45, 19, 1, 19, 45, 100])
    s1 = interpolate.UnivariateSpline(x1, y1)

    cdef np.ndarray x2 = np.array([-100, -50, -20, 0, 20, 50, 100])
    cdef np.ndarray y2 = np.array([100, 50, 20, 1, 20, 50, 100])
    s2 = interpolate.UnivariateSpline(x2, y2)

    src.A = src.A * s2(src.A) / s1(src.A)
    src.B = src.B * s2(src.B) / s1(src.B)

    return src


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


cdef np.ndarray gaussian_kernel(int radius, float std):
    window = scipy.signal.gaussian(radius, std=std)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern


cdef np.ndarray kaiser_kernel(int radius, float beta):
    window = np.kaiser(radius, beta)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern


cdef bilateral_differences(np.ndarray source, np.ndarray filtered_image, np.ndarray W, np.ndarray thread, int radius, np.ndarray pad, float std_i, float std_s):
    """
    Perform the bilateral differences and weighting on the (i, j) neighbour.
    For multithreading purposes
    This provides the loop to run inside each thread.
    """

    cdef float distance, gs
    cdef np.ndarray gi, w, neighbour
    cdef int i, j

    for (i, j) in thread:
        neighbour = pad[radius + i: radius + i + source.shape[0],
                        radius + j: radius + j + source.shape[1]]

        distance = np.sqrt(i * i + j * j)

        gi = gaussian((neighbour - source), std_i)
        gs = gaussian(distance, std_s)

        w = gi * gs
        filtered_image += neighbour * w
        W += w

cdef np.ndarray bilateral_filter_C(np.ndarray source, int radius, float std_i, float std_s, bint parallel=1):
    """
    Optimized parallel Cython function to perform bilateral filtering
    For multithreading purposes
    This provides the thread splitting and returns the filtered image
    
    """

    cdef np.ndarray filtered_image = np.zeros_like(source).astype(float)
    cdef np.ndarray pad = np.pad(source, (radius, radius), mode="symmetric")
    cdef np.ndarray W = np.zeros_like(source).astype(float)

    cdef int num_threads = os.cpu_count()

    iseq = range(-radius, radius + 1)
    jseq = iseq

    cdef np.ndarray combi = np.transpose([np.tile(iseq, len(jseq)),
                                          np.repeat(jseq, len(iseq))])

    cdef list chunks = np.array_split(combi, num_threads)
    cdef np.ndarray w, neighbour

    cdef list processing_threads = []

    for chunk in chunks:
        if parallel == 0:
            bilateral_differences(source, filtered_image,
                                  W, chunk, radius, pad, std_i, std_s)

        else:
            p = Thread(target=bilateral_differences,
                       args=(source, filtered_image, W, chunk, radius, pad, std_i, std_s))
            p.start()
            processing_threads.append(p)

    if parallel == 1:
        for thread in processing_threads:
            thread.join()

    return np.divide(filtered_image, W)


@timeit
def bilateral_filter(source, radius, std_i, std_s, parallel=True):
    """
    Expose bilateral_filter_C Cython function to Python code
    """
    return bilateral_filter_C(source, radius, std_i, std_s, parallel)


cdef np.ndarray bessel_blur(np.ndarray src, int radius, float amount):
    """
    Blur filter using Bessel function
    """
    
    src = scipy.signal.convolve2d(src,
                                  kaiser_kernel(radius, amount),
                                  mode="same",
                                  boundary="symm"
                                  )

    return src


cdef np.ndarray gaussian_blur(np.ndarray src, int radius, float amount):
    """
    Blur filter using the Gaussian function
    """
    
    src = scipy.signal.convolve2d(src,
                                  gaussian_kernel(radius, amount),
                                  mode="same",
                                  boundary="symm"
                                  )

    return src


@timeit
def USM(src, radius, strength, amount, method="bessel"):
    """
    Unsharp mask using Bessel or Gaussian blur
    """

    if method == "bessel":
        src = src + (src - bessel_blur(src, radius, strength)) * amount

    if method == "gauss":
        src = src + (src - gaussian_blur(src, radius, strength)) * amount

    return src


cdef np.ndarray overlay(np.ndarray upx, np.ndarray lpx):
    """
    Overlay blending mode between 2 layers : upx (top) and lpx (bottom)
    """

    return [lpx < 50] * (2 * upx * lpx / 100) + [lpx > 50] * \
        (100 - 2 * (100 - upx) * (100 - lpx) / 100)


@timeit
def blending(np.ndarray upx, np.ndarray lpx, str type):
    """
    Expose the blending modes to Python code
    upx : top layer
    dpx: bottom layer
    """

    types = {"overlay": overlay}

    return types[type](upx, lpx)
