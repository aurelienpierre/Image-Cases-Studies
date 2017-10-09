'''
Created on 27 avr. 2017

@author: aurelien

Source : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3166524/
'''

from scipy import interpolate
from sympy import *
from sympy.matrices import *
from threading import Thread
import numba
import os
import scipy.signal
import scipy.sparse
import time
import warnings

import numpy as np

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


def Lagrange_interpolation(points, variable=None):
    """
    Compute the Lagrange interpolation polynomial.

    :var points: A numpy n×2 ndarray of the interpolations points
    :var variable: None, float or ndarray
    :returns:   * P the symbolic expression
                * Y the evaluation of the polynomial if `variable` is float or ndarray

    """

    x = Symbol("x")
    L = zeros(1, points.shape[0])
    i = 0

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

def grey_point(src, amount):
    """
    Adjust the grey point to the desired amount using Lagrange interpolation

    """

    set1 = np.array([  # Quadratic
        [0, 1],
        [amount, amount],
        [100, 100]
    ])

    set2 = np.array([  # Quadratic
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


def auto_vibrance(src):
    """
    Add some more saturation preserving the skin tones (A < 20, B < 18)
    """
    x1 = np.array([-100, -50, -20, 0, 20, 50, 100])
    y1 = np.array([100,  45, 19, 1, 19, 45, 100])
    s1 = interpolate.UnivariateSpline(x1, y1)

    x2 = np.array([-100, -50, -20, 0, 20, 50, 100])
    y2 = np.array([100, 50, 20, 1, 20, 50, 100])
    s2 = interpolate.UnivariateSpline(x2, y2)

    src.A = src.A * s2(src.A) / s1(src.A)
    src.B = src.B * s2(src.B) / s1(src.B)

    return src


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def gaussian_kernel(radius, std):
    window = scipy.signal.gaussian(radius, std=std)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern


def kaiser_kernel(radius, beta):
    window = np.kaiser(radius, beta)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern


def poisson_kernel(radius, tau):
    window = scipy.signal.exponential(radius, tau=tau)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern


def bilateral_differences(source, filtered_image, W, thread, radius, pad, std_i, std_s):
    """
    Perform the bilateral differences and weighting on the (i, j) neighbour.
    For multithreading purposes
    This provides the loop to run inside each thread.
    """

    for (i, j) in thread:
        neighbour = pad[radius + i: radius + i + source.shape[0],
                        radius + j: radius + j + source.shape[1]]

        distance = np.sqrt(i * i + j * j)

        gi = gaussian((neighbour - source), std_i)
        gs = gaussian(distance, std_s)

        w = gi * gs
        filtered_image += neighbour * w
        W += w


@timeit
def bilateral_filter(source, radius, std_i, std_s, parallel=1):
    """
    Optimized parallel Cython function to perform bilateral filtering
    For multithreading purposes
    This provides the thread splitting and returns the filtered image

    """

    filtered_image = np.zeros_like(source).astype(float)
    pad = np.pad(source, (radius, radius), mode="symmetric")
    W = np.zeros_like(source).astype(float)

    num_threads = os.cpu_count()

    iseq = range(-radius, radius + 1)
    jseq = iseq

    combi = np.transpose([np.tile(iseq, len(jseq)),
                                    np.repeat(jseq, len(iseq))])

    chunks = np.array_split(combi, num_threads)

    processing_threads = []

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


def bessel_blur(src, radius, amount):
    """
    Blur filter using Bessel function
    """

    src = scipy.signal.convolve2d(src,
                                  kaiser_kernel(radius, amount),
                                  mode="same",
                                  boundary="symm"
                                  )

    return src


def gaussian_blur(src, radius, amount):
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
@numba.vectorize
def USM(src, radius, strength, amount, method="bessel"):
    """
    Unsharp mask using Bessel or Gaussian blur
    """

    blur = {"bessel": bessel_blur, "gauss": gaussian_blur}

    src = src + (src - blur[method](src, radius, strength)) * amount

    return src


def overlay(upx, lpx):
    """
    Overlay blending mode between 2 layers : upx (top) and lpx (bottom)
    """

    return [lpx < 50] * (2 * upx * lpx / 100) + [lpx > 50] * \
        (100 - 2 * (100 - upx) * (100 - lpx) / 100)


def blending(upx, lpx, type):
    """
    Expose the blending modes to Python code
    upx : top layer
    dpx: bottom layer
    """

    types = {"overlay": overlay}

    return types[type](upx, lpx)