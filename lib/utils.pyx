'''
Created on 27 avr. 2017

@author: aurelien

Source : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3166524/
'''
from PIL import Image
from scipy import interpolate
from scipy.signal import fftconvolve, convolve
from skimage import color
from sympy import *
from sympy.matrices import *
from threading import Thread
from multiprocessing import Pool
import os
import scipy.signal
import time
import warnings
import numba

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

def histogram(np.ndarray[DTYPE_t, ndim=3] src):
    """
    Fit the histogram entirely
    """
    delta = 100 / (np.max(src[..., 0]) - np.min(src[..., 0]))
    src[..., 0] = (src[..., 0] - np.min(src[..., 0])) * delta
    src[..., 1] = src[..., 1] * delta
    src[..., 2] = src[..., 2] * delta
    return src


def grey_point(np.ndarray[DTYPE_t, ndim=3] src, float amount):
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


def auto_vibrance(np.ndarray[DTYPE_t, ndim=2] src):
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

def gaussian(x, float sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def gaussian_kernel(int radius, float std):
    window = scipy.signal.gaussian(radius, std=std)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern

def kaiser_kernel(int radius, float beta):
    window = np.kaiser(radius, beta)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern

def poisson_kernel(int radius, float tau):
    window = scipy.signal.exponential(radius, tau=tau)
    kern = np.outer(window, window)
    kern = kern / kern.sum()
    return kern


cdef np.ndarray[DTYPE_t, ndim=2] bilateral_differences(np.ndarray[DTYPE_t, ndim=2] source,\
                                                       np.ndarray[DTYPE_t, ndim=2] filtered_image, \
                                                       np.ndarray[DTYPE_t, ndim=2] W, \
                                                       np.ndarray[long, ndim=2] thread, \
                                                       int radius, \
                                                       np.ndarray[DTYPE_t, ndim=2] pad, \
                                                       float std_i, \
                                                       float std_s):
    """
    Perform the bilateral differences and weighting on the (i, j) neighbour.
    For multithreading purposes
    This provides the loop to run inside each thread.
    """

    cdef float distance, gs
    cdef np.ndarray[DTYPE_t, ndim=2] gi, w, neighbour
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


@timeit
def bilateral_filter(np.ndarray[DTYPE_t, ndim=2] source,\
                    int radius, \
                    float std_i, \
                    float std_s, \
                    bint parallel=1):
    """
    Optimized parallel Cython function to perform bilateral filtering
    For multithreading purposes
    This provides the thread splitting and returns the filtered image
    
    """

    cdef np.ndarray[DTYPE_t, ndim=2] filtered_image = np.zeros_like(source).astype(float)
    cdef np.ndarray[DTYPE_t, ndim=2] pad = np.pad(source, (radius, radius), mode="symmetric")
    cdef np.ndarray[DTYPE_t, ndim=2] W = np.zeros_like(source).astype(float)

    cdef int num_threads = os.cpu_count()

    iseq = range(-radius, radius + 1)
    jseq = iseq

    cdef np.ndarray combi = np.transpose([np.tile(iseq, len(jseq)),
                                          np.repeat(jseq, len(iseq))])

    cdef list chunks = np.array_split(combi, num_threads)
    cdef np.ndarray[DTYPE_t, ndim=2] w, neighbour
    cdef np.ndarray[long, ndim=2] chunk

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


cdef np.ndarray[DTYPE_t, ndim=2] bessel_blur(np.ndarray[DTYPE_t, ndim=2] src,\
                                             int radius, float amount):
    """
    Blur filter using Bessel function
    """
    
    src = scipy.signal.convolve2d(src,
                                  kaiser_kernel(radius, amount),
                                  mode="same",
                                  boundary="symm"
                                  )

    return src


cdef np.ndarray[DTYPE_t, ndim=2] gaussian_blur(np.ndarray[DTYPE_t, ndim=2] src,\
                                               int radius, float amount):
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


cdef np.ndarray[DTYPE_t, ndim=2] overlay(np.ndarray[DTYPE_t, ndim=2] upx,\
                                         np.ndarray[DTYPE_t, ndim=2] lpx):
    """
    Overlay blending mode between 2 layers : upx (top) and lpx (bottom)
    """

    return [lpx < 50] * (2 * upx * lpx / 100) + [lpx > 50] * \
        (100 - 2 * (100 - upx) * (100 - lpx) / 100)


@timeit
def blending(np.ndarray[DTYPE_t, ndim=2] upx,\
             np.ndarray[DTYPE_t, ndim=2] lpx, \
             str type):
    """
    Expose the blending modes to Python code
    upx : top layer
    dpx: bottom layer
    """

    types = {"overlay": overlay}

    return types[type](upx, lpx)


def richardson_lucy_2d(np.ndarray[DTYPE_t, ndim=2] image,\
                    np.ndarray[DTYPE_t, ndim=2] psf, \
                    float damp, \
                    int iterations=50, \
                    str mode="LAB"):
    """Richardson-Lucy deconvolution.
     
    Adapted for Cython and performance from skimage.restore module
 
    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    damp : float
        Removes the pixels that are `damp` times the standard deviations further
        from the original image. This prevent noise amplification. Set damp = 0
        to bypass noise damping.
 
 
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
     
    # Pad the image with symmetric data to avoid border effects
    image = np.pad(image, (iterations, iterations), mode="symmetric")
     
    cdef np.ndarray[DTYPE_t, ndim=2] im_deconv = 0.5 * np.ones_like(image)
    cdef np.ndarray[DTYPE_t, ndim=2] psf_mirror = psf[::-1, ::-1]
    cdef np.ndarray[DTYPE_t, ndim=2] relative_blur, im_backup, delta, 
    cdef list damping_array
 
 
    # There is a way to make it a recursive function, which is more elegant, but it my tests it was a bit slower
    for _ in range(iterations):
        relative_blur = image / fftconvolve(im_deconv, psf, 'same')
        im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')
        
        if damp != 0:
            # Remove the current iteration for pixels where the difference
            # between original and deconvoluted image is above damp * std
            # this prevents noise amplification from one iteration to another
            delta = np.absolute(image - im_deconv)
            damping_array = [delta > damp * delta.std()]
            im_deconv[damping_array] = image[damping_array]
         
    image = im_deconv[iterations:-iterations, iterations:-iterations]
     
    return image 
 
 
cdef richardson_lucy_3d(np.ndarray[DTYPE_t, ndim=3] image,\
                    np.ndarray[DTYPE_t, ndim=2] psf, \
                    float damp, \
                    int iterations=50, \
                    str mode="LAB"):
    """
    Run 3 Richardson-Lucy deconvolutions on 3 channels in 3 different threads
    """
     
    cdef list processing_threads = []
    cdef int i
     
    with Pool(processes=3) as pool:
 
        image = np.dstack(pool.starmap(
                                        richardson_lucy_2d, 
                                        [(image[..., i], psf, damp, iterations, mode) for i in range(3)]
                                        )
                          )
 
         
    return image

@timeit
def richardson_lucy(np.ndarray image,\
                    np.ndarray[DTYPE_t, ndim=2] psf, \
                    float damp, \
                    int iterations=50,\
                    str mode="LAB"):

    """Expose C function to Python"""
    
    if image.ndim == 3:
        return richardson_lucy_3d(image, psf, damp, iterations, mode)
    
    if image.ndim == 2:
        return richardson_lucy_2d(image, psf, damp, iterations, mode)
