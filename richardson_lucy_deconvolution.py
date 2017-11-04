'''
Created on 30 avr. 2017

@author: aurelien

This script shows an implementation of the Richardson-Lucy deconvolution.

In theory, blurred and noisy pictures can be perfectly sharpened if we perfectly
know the [*Point spread function*](https://en.wikipedia.org/wiki/Point_spread_function)
of their maker. In practice, we can only estimate it.
One of the means to do so is the [Richardson-Lucy deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution).

The Richardson-Lucy algorithm used here has a damping coefficient wich allows to remove from
the iterations the pixels which deviate to much (x times the standard deviation of the difference
source image - deconvoluted image) from the original image. This pixels are considered
noise and would be amplificated from iteration to iteration otherwise.
'''

import multiprocessing
import warnings
from os.path import join

import numpy as np
import scipy
from PIL import Image
from numba import float32, jit, int16, boolean, int8, prange
from scipy import ndimage
from scipy.signal import convolve
from skimage.restoration import denoise_tv_chambolle

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack

except:
    pass

warnings.simplefilter("ignore", (DeprecationWarning, UserWarning))

from lib import utils

CPU = 8


@jit(float32[:](float32[:], float32, float32), cache=True, nogil=True)
def best_epsilon(image, lambd, p=0.5):
    """
    Find the minimum acceptable epsilon to avoid a degenerate constant solution

    Reference : http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf
    :param image:
    :param lambd:
    :param p:
    :return:
    """
    grad_image = np.gradient(image, edge_order=2)
    norm_grad_image = np.sqrt(grad_image[0] ** 2 + grad_image[1] ** 2)
    omega = 2 * lambd * np.amax(image - image.mean()) / (p * image.size)
    epsilon = np.sqrt(norm_grad_image.mean() / (np.exp(omega) - 1))
    return np.maximum(epsilon, 1e-31)


@jit(float32[:](float32[:]), cache=True)
def divTV(image):
    """Compute the Total Variation norm

    :param ndarray image: Input array of pixels
    :return: div(Total Variation regularization term) as described in [3]
    :rtype: ndarray

    """
    grad = np.zeros_like(image)

    # Forward differences
    # fx = np.roll(image, 1, axis=1) - image
    fx = np.pad(image, ((0, 0), (1, 0)), mode="edge")[:, 1:] - image
    # fy = np.roll(image, 1, axis=0) - image
    fy = np.pad(image, ((1, 0), (0, 0)), mode="edge")[1:, :] - image
    grad += (fx + fy) / np.maximum(1e-3, np.sqrt(fx ** 2 + fy ** 2))

    # Backward x and crossed y differences
    # fx = image - np.roll(image, -1, axis=1)
    fx = np.pad(image, ((0, 0), (0, 1)), mode="edge")[:, :-1] - image
    # fy = np.roll(image, (-1, 1), axis=(0, 1)) - np.roll(image, -1, axis=0)
    fy = np.pad(image, ((0, 1), (1, 0)), mode="edge")[:-1, 1:] - np.pad(image, ((1, 0), (0, 0)), mode="edge")[1:, :]
    grad -= fx / np.maximum(1e-3, np.sqrt(fx ** 2 + fy ** 2))

    # Backward y and crossed x differences
    # fy = image - np.roll(image, -1, axis=0)
    fy = np.pad(image, ((0, 1), (0, 0)), mode="edge")[:-1, :] - image
    # fx = np.roll(image, (1, -1), axis=(0, 1)) - np.roll(image, -1, axis=1)
    fx = np.pad(image, ((1, 0), (0, 1)), mode="edge")[1:, :-1] - np.pad(image, ((0, 0), (0, 1)), mode="edge")[:, 1:]
    grad -= fy / np.maximum(1e-3, np.sqrt(fy ** 2 + fx ** 2))

    return grad.astype(np.float32)


@jit(float32[:](float32[:], float32, float32, float32, float32), cache=True, nogil=True)
def center_diff(u, dx, dy, epsilon, p):
    # Centered local difference
    ux = np.roll(u, (dx, 0), axis=(1, 0)) - np.roll(u, (0, 0), axis=(1, 0))
    uy = np.roll(u, (0, dy)) - np.roll(u, (0, 0))
    TV = (np.abs(ux) ** p + np.abs(uy) ** p + epsilon) ** (1 / p)
    du = - ux - uy

    return TV, du


@jit(float32[:](float32[:], float32, float32, float32, float32), cache=True, nogil=True)
def x_diff(u, dx, dy, epsilon, p):
    # x-shifted local difference
    ux = np.roll(u, (0, 0), axis=(1, 0)) - np.roll(u, (-dx, 0), axis=(1, 0))
    uy = np.roll(u, (-dx, dy), axis=(1, 0)) - np.roll(u, (-dx, 0), axis=(1, 0))
    TV = (np.abs(ux) ** p + np.abs(uy) ** p + epsilon) ** (1 / p)
    du = ux

    return TV, du


@jit(float32[:](float32[:], float32, float32, float32, float32), cache=True, nogil=True)
def y_diff(u, dx, dy, epsilon, p):
    # y shifted local difference
    ux = np.roll(u, (dx, -dy), axis=(1, 0)) - np.roll(u, (0, -dy), axis=(1, 0))
    uy = np.roll(u, (0, 0), axis=(1, 0)) - np.roll(u, (0, -dy), axis=(1, 0))
    TV = (np.abs(ux) ** p + np.abs(uy) ** p + epsilon) ** (1 / p)
    du = uy

    return TV, du


@jit(float32[:](float32[:], float32[:], float32, float32, float32), cache=True, nogil=True)
def gradTVEM(u, ut, epsilon=1e-3, tau=1e-1, p=0.5):
    """Compute the Total Variation norm of the Minimization-Maximization problem

    :param ndarray image: Input array of pixels
    :return: div(Total Variation regularization term) as described in [3]
    :rtype: ndarray

    We use general P-norm instead : https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-153.pdf

    0.5-norm shows better representation of discontinuities : https://link.springer.com/chapter/10.1007/978-3-319-14612-6_10

    """

    # Displacement vectors of the shifted differences
    deltas = np.array([[1, 1],
                       [-1, 1],
                       [1, -1],
                       [-1, -1]])

    # 2-axis shifts
    u_copy = np.zeros_like(u)
    shifts = np.array([u_copy,  # Centered
                       u_copy,  # x shifted
                       u_copy  # y shifted
                       ])

    # Methods for local differences calculation
    diffs = [center_diff, x_diff, y_diff]

    # Initialization of the outputs
    du = np.array([shifts, shifts, shifts, shifts])
    TV = du.copy()
    TVt = du.copy()

    dx = 0
    dy = 0

    for i in prange(4):
        # for each displacement vector
        dx = deltas[i, 0]
        dy = deltas[i, 1]

        for step in prange(3):
            # for each axial shift
            TV[i, step], du[i, step] = diffs[step](u, dy, dy, epsilon, p)
            TVt[i, step], void = diffs[step](ut, dx, dy, epsilon, p)

    grad = np.sum(du / TV / (tau + TVt), axis=(0, 1))  # This is the vectorized equivalent to :

    """
    grad = np.zeros_like(u)
    for channel in prange(3):
        for delta in range(4):
            for direction in range(3):
                grad +=  du[delta, direction] / \ 
                                         TV[delta, direction] /\
                                                          (TVt[delta, direction] + tau)
    """

    return grad / 4

def pad_image(image: np.ndarray, pad: tuple, mode="edge"):
    """
    Pad an 3D image with a free-boundary condition to avoid ringing along the borders after the FFT

    :param image:
    :param pad:
    :param mode:
    :return:
    """
    R = np.pad(image[..., 0], pad, mode=mode)
    G = np.pad(image[..., 1], pad, mode=mode)
    B = np.pad(image[..., 2], pad, mode=mode)
    u = np.dstack((R, G, B))
    return np.ascontiguousarray(u, np.float32)


def unpad_image(image: np.ndarray, pad: tuple):
    return np.ascontiguousarray(image[pad[0]:-pad[0], pad[1]:-pad[1], ...], np.ndarray)


@jit(cache=True)
def build_pyramid(psf_size: int, lambd: float, method) -> list:
    """
    To speed-up the deconvolution, the PSF is estimated successively on smaller images of increasing sizes. This function
    computes the intermediates sizes and regularization factors

    :param image_size:
    :param psf_size:
    :param lambd:
    :param method:
    :return:
    """
    lambdas = [lambd]
    images = [1]
    kernels = [psf_size]

    if method == richardson_lucy_PAM:
        image_multiplier = 1.1
        lambda_multiplier = 1.9
        lambda_max = 0.5

    elif method == richardson_lucy_MM:
        image_multiplier = np.sqrt(2)
        lambda_multiplier = 1 / 2.1
        lambda_max = 999999

    while kernels[-1] > 3:
        lambdas.append(min([lambdas[-1] * lambda_multiplier, lambda_max]))
        kernels.append(int(np.floor(kernels[-1] / image_multiplier)))
        images.append(images[-1] / image_multiplier)

        if kernels[-1] % 2 == 0:
            kernels[-1] -= 1

        if kernels[-1] < 3:
            kernels[-1] = 3

    print(kernels)

    return images, kernels, lambdas


def process_pyramid(pic, u, psf, lambd, method, epsilon=1e-3, quality=1):
    """
    To speed-up the deconvolution, the PSF is estimated successively on smaller images of increasing sizes. This function
    computes the intermediates deblured pictures and PSF.

    :param pic:
    :param u:
    :param psf:
    :param lambd:
    :param method:
    :param epsilon:
    :param quality: the number of iterations performed at each pyramid step will be adjusted by this factor.
    :return:
    """
    Mk, Nk, C = psf.shape
    images, kernels, lambdas = build_pyramid(Mk, lambd, method)
    u = ndimage.zoom(u, (images[-1], images[-1], 1))
    k_prec = Mk

    for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):
        print("== Pyramid step", i, "==")

        # Resize blured, deblured images and PSF from previous step
        if i != 1:
            im = ndimage.zoom(pic, (i, i, 1))
        else:
            im = pic.copy()

        psf = ndimage.zoom(psf, (k / k_prec, k / k_prec, 1))
        psf = _normalize_kernel(psf)
        u = ndimage.zoom(u, (im.shape[0] / u.shape[0], im.shape[1] / u.shape[1], 1))

        # Make a blind Richardson-Lucy deconvolution on the RGB signal
        iterations = int(k ** 2 * quality)
        u, psf = method(im, u, psf, l, iterations, epsilon=epsilon)

        k_prec = k

    return u, psf


def make_preview(image, psf, ratio, mask=None):
    """
    Resize the image, the PSF and the mask to preview the settings on a smaller picture to speed-up the tweaking

    :param image:
    :param psf:
    :param ratio:
    :param mask:
    :return:
    """
    image = ndimage.zoom(image, (ratio, ratio, 1))

    MK_source = psf.shape[0]
    MK = int(MK_source * ratio)

    if MK % 2 == 0:
        MK += 1

    if MK < 3:
        MK = 3

    psf = _normalize_kernel(ndimage.zoom(psf, (MK / MK_source, MK / MK_source, 1)))

    if mask:
        mask = [int(x * ratio) for x in mask]

    return image, psf, mask


@jit(float32[:](float32[:]), cache=True, nogil=True)
def _normalize_kernel(kern):
    kern[kern < 0] = 0
    kern /= np.sum(kern, axis=(0, 1))
    return kern


@jit(float32[:](float32[:], float32[:], float32[:]), cache=True, nogil=True)
def _convolve_image(u, image, psf):
    error = np.ascontiguousarray(convolve(u, psf, "valid"), np.float32)
    error -= image
    return convolve(error, np.rot90(psf, 2), "full")


@jit(float32[:](float32[:], float32[:], float32[:]), cache=True, nogil=True)
def _convolve_kernel(u, image, psf):
    error = np.ascontiguousarray(convolve(u, psf, "valid"), np.float32)
    error -= image
    return convolve(np.rot90(u, 2), error, "valid")


@jit(float32[:](float32[:], float32[:], float32[:], float32, float32), cache=True, nogil=True)
def _update_image_PAM(u, image, psf, lambd, epsilon=5e-3):
    gradu, TV = divTV(u)
    gradu /= TV
    gradu *= lambd
    gradu += _convolve_image(u, image, psf)
    weight = epsilon * np.amax(u) / np.amax(np.abs(gradu))
    u -= weight * gradu
    return u


@jit(float32[:](float32[:], float32[:], float32[:], float32, int16, float32), cache=True, nogil=True)
def _loop_update_image_PAM(u, image, psf, lambd, iterations, epsilon):
    for itt in range(iterations):
        u = _update_image_PAM(u, image, psf, lambd, epsilon)
        lambd *= 0.99
    return u, psf


@jit(float32[:](float32[:], float32[:], float32[:], float32), cache=True, nogil=True)
def _update_kernel_PAM(u, image, psf, epsilon):
    grad_psf = _convolve_kernel(u, image, psf)
    weight = epsilon * np.amax(psf) / np.maximum(1e-31, np.amax(np.abs(grad_psf)))
    psf -= weight * grad_psf
    psf = _normalize_kernel(psf)
    return psf


@jit(float32[:](float32[:], float32[:], float32[:], float32, int16, float32[:], float32[:], float32), cache=True,
     nogil=True)
def _loop_update_both_PAM(u, image, psf, lambd, iterations, mask_u, mask_i, epsilon):
    """
    Utility function to launch the actual deconvolution in parallel
    :param u:
    :param image:
    :param psf:
    :param lambd:
    :param iterations:
    :param mask_u:
    :param mask_i:
    :param epsilon:
    :return:
    """
    for itt in range(iterations):
        u = _update_image_PAM(u, image, psf, lambd, epsilon)
        psf = _update_kernel_PAM(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3]],
                                 image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3]], psf, epsilon)
        lambd *= 0.99
    return u, psf


@utils.timeit
def richardson_lucy_PAM(image: np.ndarray,
                        u: np.ndarray,
                        psf: np.ndarray,
                        lambd: float,
                        iterations: int,
                        epsilon=1e-3,
                        mask=None,
                        blind=True) -> np.ndarray:
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization by Projected Alternating Minimization.
    This is known to give a close-enough sharp image but never give an accurate sharp image.

    Based on Matlab sourcecode of D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2014: http://www.cvg.unibe.ch/dperrone/tvdb/

    :param ndarray image : Input 3 channels image.
    :param ndarray psf : The point spread function.
    :param int iterations : Number of iterations.
    :param float lambd : Lambda parameter of the total Variation regularization
    :param bool blind : Determine if it is a blind deconvolution is launched, thus if the PSF is updated
        between two iterations
    :returns ndarray: deconvoluted image

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cvg.unibe.ch/dperrone/tvdb/perrone2014tv.pdf
    .. [3] http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf
    .. [4] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [5] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    """

    # image dimensions
    MK, NK, CK = psf.shape
    M, N, C = image.shape

    # Verify the input and scream like a virgin
    assert (CK == C), "Dimensions of the PSF and of the image don't match !"
    assert (MK == NK), "The PSF must be square"
    assert (MK >= 3), "The dimensions of the PSF are too small !"
    assert (M > MK and N > NK), "The size of the picture is smaller than the PSF !"
    assert (MK % 2 != 0), "The dimensions of the PSF must be odd !"

    # Prepare the picture for FFT convolution by padding it with pixels that will be removed
    pad = np.floor(MK / 2).astype(int)
    u = pad_image(u, (pad, pad))

    print("working on image :", u.shape)

    # Adjust the coordinates of the masks with the padding dimensions
    if mask == None:
        mask_i = [0, M + 2 * pad, 0, N + 2 * pad]
        mask_u = [0, M + 2 * pad, 0, N + 2 * pad]
    else:
        mask_u = [mask[0] - pad, mask[1] + pad, mask[2] - pad, mask[3] + pad]
        mask_i = mask

    # Start 3 parallel processes
    pool = multiprocessing.Pool(processes=CPU)

    if blind:
        # Blind deconvolution with PSF refinement
        output = pool.starmap(_loop_update_both_PAM,
                              [(u[..., 0], image[..., 0], psf[..., 0], lambd, iterations, mask_u, mask_i, epsilon),
                               (u[..., 1], image[..., 1], psf[..., 1], lambd, iterations, mask_u, mask_i, epsilon),
                               (u[..., 2], image[..., 2], psf[..., 2], lambd, iterations, mask_u, mask_i, epsilon),
                               ]
                              )

        u = np.dstack((output[0][0], output[1][0], output[2][0]))
        psf = np.dstack((output[0][1], output[1][1], output[2][1]))

    else:
        # Regular deconvolution without PSF refinement
        output = pool.starmap(_loop_update_image_PAM,
                              [(u[..., 0], image[..., 0], psf[..., 0], lambd, iterations, epsilon),
                               (u[..., 1], image[..., 1], psf[..., 1], lambd, iterations, epsilon),
                               (u[..., 2], image[..., 2], psf[..., 2], lambd, iterations, epsilon),
                               ]
                              )

        u = np.dstack((output[0][0], output[1][0], output[2][0]))
        psf = np.dstack((output[0][1], output[1][1], output[2][1]))

    print(iterations, "iterations")
    u = unpad_image(u, (pad, pad))
    pool.close()

    return u.astype(np.float32), psf


@jit(float32[:](float32[:], float32[:], float32[:], float32, int16, float32[:], float32[:], float32), cache=True,
     nogil=True)
def _update_both_MM(u, image, psf, lambd, iterations, mask_u, mask_i, epsilon):
    """
    Utility function to launch the actual deconvolution in parallel
    :param u:
    :param image:
    :param psf:
    :param lambd:
    :param iterations:
    :param mask_u:
    :param mask_i:
    :param epsilon:
    :return:
    """
    tau = 1e-3
    lambd = 1 / lambd

    k_step = 1e-3
    u_step = 1e-3

    for it in range(iterations):
        ut = u
        for itt in range(5):
            # Image update
            epsilon = best_epsilon(u, lambd) * 1.001
            gradu = lambd * _convolve_image(u, image, psf) + utils.gradTVEM(u, ut, epsilon, tau)
            dt = u_step * (np.amax(u) + 1 / u.size) / np.amax(np.abs(gradu) + 1e-31)
            u -= dt * gradu

            # PSF update
            gradk = _convolve_kernel(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3]],
                                     image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3]], psf)
            alpha = k_step * (np.amax(psf) + 1 / psf.size) / np.amax(np.abs(gradk) + 1e-31)
            psf -= alpha * gradk
            psf = _normalize_kernel(psf)

        lambd *= 1.001

    return u.astype(np.float32), psf


@utils.timeit
def richardson_lucy_MM(image: np.ndarray,
                       u: np.ndarray,
                       psf: np.ndarray,
                       lambd: float,
                       iterations: int,
                       epsilon=5e-3,
                       mask=None,
                       blind=True) -> np.ndarray:
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization by the Minimization-Maximization
    algorithm. This is known to give the sharp image in more than 50 % of the cases.

    Based on Matlab sourcecode of :

    Copyright (C) Daniele Perrone, perrone@iam.unibe.ch
	      Remo Diethelm, remo.diethelm@outlook.com
	      Paolo Favaro, paolo.favaro@iam.unibe.ch
	      2014, All rights reserved.

    :param ndarray image : Input 3 channels image.
    :param ndarray psf : The point spread function.
    :param int iterations : Number of iterations.
    :param float lambd : Lambda parameter of the total Variation regularization
    :param bool blind : Determine if it is a blind deconvolution is launched, thus if the PSF is updated
        between two iterations
    :returns ndarray: deconvoluted image

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cvg.unibe.ch/dperrone/tvdb/perrone2014tv.pdf
    .. [3] http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf
    .. [4] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [5] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    """

    # image dimensions
    MK, NK, C = psf.shape
    M, N, C = image.shape
    pad = np.floor(MK / 2).astype(int)

    print("working on image :", u.shape)

    u = pad_image(u, (pad, pad))

    if mask == None:
        mask_i = [0, M + 2 * pad, 0, N + 2 * pad]
        mask_u = [0, M + 2 * pad, 0, N + 2 * pad]
    else:
        mask_u = [mask[0] - pad, mask[1] + pad, mask[2] - pad, mask[3] + pad]
        mask_i = mask

    iterations = int(np.maximum(iterations / 5, 2))

    try:
        pool = multiprocessing.Pool(processes=CPU)
        output = pool.starmap(_update_both_MM,
                              [(u[..., 0], image[..., 0], psf[..., 0], lambd, iterations, mask_u, mask_i, epsilon),
                               (u[..., 1], image[..., 1], psf[..., 1], lambd, iterations, mask_u, mask_i, epsilon),
                               (u[..., 2], image[..., 2], psf[..., 2], lambd, iterations, mask_u, mask_i, epsilon),
                               ]
                              )

        u = np.dstack((output[0][0], output[1][0], output[2][0]))
        psf = np.dstack((output[0][1], output[1][1], output[2][1]))
        pool.close()

    except MemoryError:
        pool = multiprocessing.Pool(processes=1)

        output = pool.starmap(_update_both_MM,
                              [(u[..., 0], image[..., 0], psf[..., 0], lambd, iterations, mask_u, mask_i, epsilon),
                               (u[..., 1], image[..., 1], psf[..., 1], lambd, iterations, mask_u, mask_i, epsilon),
                               (u[..., 2], image[..., 2], psf[..., 2], lambd, iterations, mask_u, mask_i, epsilon),
                               ]
                              )

        u = np.dstack((output[0][0], output[1][0], output[2][0]))
        psf = np.dstack((output[0][1], output[1][1], output[2][1]))
        pool.close()

    u = unpad_image(u, (pad, pad))
    print(iterations * 5, "iterations")
    return u.astype(np.float32), psf


@utils.timeit
@jit(float32[:](float32[:],
                int8,
                int8,
                int8,
                int16,
                float32,
                float32,
                int16,
                float32,
                float32,
                boolean,
                int16,
                int16[:],
                float32,
                boolean,
                float32,
                float32,
                float32[:],
                boolean,
                int8),
     parallel=True)
def deblur_module(pic: np.ndarray,
                  filename: str,
                  dest_path: str,
                  blur_type: str,
                  blur_width: int,
                  noise_damping: float,
                  deblur_strength: int,
                  blur_strength: int = 1,
                  auto_quality=1,
                  epsilon=1e-3,
                  refine: bool = False,
                  refine_quality=0,
                  mask: np.ndarray = None,
                  backvsmask_ratio: float = 0,
                  debug: bool = False,
                  effect_strength=1,
                  preview=1,
                  psf: np.ndarray = None,
                  denoise: bool = False,
                  method="fast"):
    """
    This mimics a Darktable module inputs where the technical specs are hidden and translated into human-readable parameters.

    It's an interface between the regular user and the geeky actual deconvolution parameters

    :param pic: Blured image in RGB 8 bits as a 3D array where the last dimension is the RGB channel
    :param filename: File name to use to save the deblured picture
    :param destpath: The destination path to save the picture
    :param blur_type: kind of blur or "auto" to perform a blind deconvolution. Use "auto" for motion blur or composite blur.
    Other parameters : `poisson`, `gaussian`, `kaiser`
    :param blur_width: the width of the blur in px - must be an odd integer
    :param blur_strength: the strength of the blur, thus the standard deviation of the blur kernel
    :param auto_quality: when the `blur_type` is `auto`, the number of iterations of the initial blur estimation is half the
    square of the PSF size. The `auto_quality` parameter is a factor that allows you to reduce the number of iteration to speed up the process.
    Default : 1. Recommended values : between 0.25 and 2.
    :param noise_damping: the noise  reduction factor lambda. Typically between 0.00003 and 0.5. Increase it if smudges or noise appear
    :param refine: True or False, decide if the blur kernel should be refined through myopic deconvolution
    :param refine_quality: the number of iterations to perform during the refinement step
    :param mask: the coordinates of the rectangular mask to apply on the image to refine the blur kernel from the top-left corner of the image
        in list [y_top, y_bottom, x_left, x_right]
    :param backvsmask_ratio: when a mask is used, the ratio  of weights of the whole image / the masked zone.
        0 means only the masked zone is used, 1 means the masked zone is ignored and only the whole image is taken. 0 is
        runs faster, 1 runs much slower.
    :param preview: If you want to fast preview your setting on a smaller picture size, set the downsampling ratio in `previem`. Default : 1
    :param psf: if you already know the PSF kernel, enter it here as an 3D array where the last dimension is the RGB component
    :param denoise: True or False. Perform an initial denoising by Total Variation - Chambolle algorithm before deconvoluting
    :param method: `fast` or `best`. Set the method to deconvolute.
    :return:
    """
    # TODO : refocus http://web.media.mit.edu/~bandy/refocus/PG07refocus.pdf
    # TODO : extract foreground only https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut


    # Verify the input and scream like a virgin
    assert (blur_width >= 3), "The PSF kernel is too small !"
    assert (blur_width % 2 != 0), "The dimensions of the PSF must be odd !"
    assert (backvsmask_ratio <= 1), "The background/mask ratio must be between 0 and 1"

    # Backup ICC color profile
    icc_profile = pic.info.get("icc_profile")

    # Assuming 8 bits input, we rescale the RGB values betweem 0 and 1
    pic = np.ascontiguousarray(pic, np.float32) / 255

    methods_collection = {
        "fast": richardson_lucy_PAM,
        "best": richardson_lucy_MM
    }

    richardson_lucy = methods_collection[method]

    if psf == None:
        blur_collection = {
            "gaussian": utils.gaussian_kernel(blur_width, blur_strength),
            "kaiser": utils.kaiser_kernel(blur_width, blur_strength),
            "auto": utils.uniform_kernel(blur_width),
            "poisson": utils.poisson_kernel(blur_width, blur_strength),
        }

        # TODO http://yehar.com/blog/?p=1495

        psf = blur_collection[blur_type]

    psf = np.dstack((psf, psf, psf))

    if preview != 1:
        print("\nWorking on a scaled picture")
        pic, psf, mask = make_preview(pic, psf, preview, mask)

    if denoise:
        # TODO : http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6572468
        pic = denoise_tv_chambolle(pic, weight=noise_damping, multichannel=True)

    u = pic.copy()

    iter_background = 0
    iter_mask = 0

    if blur_type == "auto":
        print("\n===== BLIND ESTIMATION OF BLUR =====")
        u, psf = process_pyramid(pic, u, psf, noise_damping, richardson_lucy, epsilon=epsilon, quality=auto_quality)

    if refine:
        if mask:
            print("\n===== BLIND MASKED REFINEMENT =====")
            iter_background = int(round(refine_quality * backvsmask_ratio))
            iter_mask = int(round(refine_quality - iter_background))

            u, psf = richardson_lucy(pic, u, psf, noise_damping, iter_mask, mask=mask, epsilon=epsilon)

            if backvsmask_ratio != 0:
                print("\n===== BLIND UNMASKED REFINEMENT =====")
                u, psf = richardson_lucy(pic, u, psf, noise_damping, iter_background, epsilon=epsilon)

        else:
            print("\n===== BLIND UNMASKED REFINEMENT =====")
            u, psf = richardson_lucy(pic, u, psf, noise_damping, iter_background, epsilon=epsilon)

    if deblur_strength > 0:
        print("\n===== REGULAR DECONVOLUTION =====")
        u, psf = richardson_lucy_PAM(pic, u, psf, noise_damping, deblur_strength, blind=False, epsilon=epsilon)

    np.clip(u, 0, 1, out=u)

    if denoise:
        # TODO : http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6572468
        u = denoise_tv_chambolle(pic, weight=noise_damping, multichannel=True)

    # Convert back into 8 bits RGB
    u = (pic - effect_strength * (pic - u)) * 255

    if debug and mask:
        # Print the mask in debug mode
        utils.save(u, filename, dest_path, mask=mask, icc_profile=icc_profile)
    else:
        utils.save(u, filename, dest_path, icc_profile=icc_profile)

    return u, psf


if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    picture = "blured.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # deblur_module(pic, "fast-v4", "kaiser", 0, 0.05, 50, blur_width=11, blur_strength=8)

        # deblur_module(pic, "myope-v4", "kaiser", 10, 0.05, 50, blur_width=11, blur_strength=8, mask=[150, 150 + 256, 600, 600 + 256], refine=True,)
        """
        deblur_module(pic, picture + "-blind-v10-best", dest_path, "auto", 5, 0.0005, 0,
                      mask=[150, 150 + 512, 600, 600 + 512],
                      refine=True,
                      refine_quality=50,
                      auto_quality=2,
                      backvsmask_ratio=0,
                      method="best",
                      epsilon=5e-1,
                      debug=True)
        """

        """
        deblur_module(pic, picture + "-blind-v10-fast", dest_path, "auto", 5, 0.05, 0,
                      mask=[150, 150 + 512, 600, 600 + 512],
                      refine=True,
                      refine_quality=50,
                      auto_quality=2,
                      backvsmask_ratio=0,
                      method="fast",
                      debug=True)
                      
        """
        pass

    picture = "Shoot-Sienna-Hayes-0042-_DSC0284-sans-PHOTOSHOP-WEB.jpg"
    with Image.open(join(source_path, picture)) as pic:
        """
        deblur_module(pic, picture + "test-v7", dest_path, "auto", 9, 0.00005, 0,
                      mask=[318, 357 + 800, 357, 357 + 440],
                      denoise=False,
                      refine=False,
                      refine_quality=50,
                      auto_quality=1,
                      backvsmask_ratio=0,
                      preview=1,
                      debug=True,
                      method="best",
                      )

        """
        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [631 + 512, 631 + 512 + 1024, 2826 + 512, 2826 + 512 + 1024]

        deblur_module(pic, picture + "test-v7-gradient-alternatif-3-2", dest_path, "auto", 13, 0.05, 0,
                      mask=mask,
                      denoise=False,
                      refine=True,
                      refine_quality=500,
                      auto_quality=1,
                      backvsmask_ratio=0,
                      preview=0.5,
                      debug=True,
                      method="best",
                      )
        pass
