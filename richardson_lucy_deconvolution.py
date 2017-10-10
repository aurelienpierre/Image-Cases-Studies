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
from os.path import join

import numpy as np
from PIL import Image
from numba import float32, int16, jit
from scipy.signal import fftconvolve

from lib import utils


@jit(float32[:, :](float32[:, :]), cache=True)
def divTV(image: np.ndarray) -> np.ndarray:
    """Compute the second-order divergence of the pixel matrix, known as the Total Variation.

    :param ndarray image: Input array of pixels
    :return: div(Total Variation regularization term) as described in [3]
    :rtype: ndarray

    References
    ----------


    """
    grad = np.array(np.gradient(image, edge_order=2))
    grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    grad = grad / np.amax(grad)
    return grad


@jit(cache=True)
def convergence(image_after: np.ndarray, image_before: np.ndarray) -> float:
    """Compute the convergence rate between 2 iterations

    :param ndarray image_after: Image @ iteration n
    :param ndarray image_before: Image @ iteration n-1
    :param int padding: Number of pixels to ignore along each side
    :return float: convergence rate
    """
    convergence = np.log(np.linalg.norm(image_after) / np.linalg.norm((image_before)))
    print("Convergence :", convergence)
    return convergence


@jit(float32[:, :](float32[:, :], float32[:, :], int16, float32[:, :]), cache=True)
def update_image(image, u, lambd, psf):
    """Update one channel only (R, G or B)

    :param image:
    :param u:
    :param lambd:
    :param psf:
    :return:
    """
    # Richardson-Lucy deconvolution
    gradUdata = fftconvolve(fftconvolve(u, psf, "valid") - image, np.rot90(psf), "full")

    # Total Variation Regularization
    gradu = gradUdata - lambd * divTV(u)
    sf = 5E-3 * np.max(u) / np.maximum(1E-31, np.amax(np.abs(gradu)))
    u = u - sf * gradu

    # Normalize for 8 bits RGB values
    u = np.clip(u, 0.0000001, 255)

    return u


@jit(float32[:, :](float32[:, :], float32[:, :], int16, float32[:, :], int16), cache=True)
def loop_update_image(image, u, lambd, psf, iterations):
    for i in range(iterations):
        # Richardson-Lucy deconvolution
        gradUdata = fftconvolve(fftconvolve(u, psf, "valid") - image, np.rot90(psf), "full")

        # Total Variation Regularization
        gradu = gradUdata - lambd * divTV(u)
        sf = 5E-3 * np.max(u) / np.maximum(1E-31, np.amax(np.abs(gradu)))
        u = u - sf * gradu

        # Normalize for 8 bits RGB values
        u = np.clip(u, 0.0000001, 255)

        lambd = lambd / 2

    return u

@jit(float32[:, :](float32[:, :], int16, int16), cache=True)
def pad_image(image: np.ndarray, pad_v: int, pad_h: int):
    R = np.pad(image[..., 0], (pad_v, pad_h), mode="edge")
    G = np.pad(image[..., 1], (pad_v, pad_h), mode="edge")
    B = np.pad(image[..., 2], (pad_v, pad_h), mode="edge")
    u = np.dstack((R, G, B))
    return u


@jit(float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :]), cache=True)
def update_kernel(gradk: np.ndarray, u: np.ndarray, psf: np.ndarray, image: np.ndarray) -> np.ndarray:
    gradk = gradk + fftconvolve(np.rot90(u, 2), fftconvolve(u, psf, "valid") - image, "valid")

    sh = 1e-3 * np.amax(psf) / np.maximum(1e-31, np.amax(np.abs(gradk)))
    psf = psf - sh * gradk
    psf = psf / np.sum(psf)

    return psf, gradk


# @jit(float32[:,:](float32[:,:], float32[:,:], float32, int16), cache=True)
def richardson_lucy(image: np.ndarray, psf: np.ndarray, lambd: float, iterations: int,
                    blind: bool = False) -> np.ndarray:
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization.

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
    M, N, C = image.shape
    MK, NK = psf.shape
    pad_v = np.floor(MK / 2).astype(int)
    pad_h = np.floor(NK / 2).astype(int)

    # Pad the image on each channel with data to avoid border effects
    u = pad_image(image, pad_v, pad_h)

    if blind:
        gradk = np.zeros((MK, NK))

        for i in range(iterations):

            # Update sharp image
            with multiprocessing.Pool(processes=3) as pool:
                u = np.dstack(pool.starmap(
                    update_image,
                    [(image[..., chan], u[..., chan], lambd, psf) for chan in range(C)]
                )
                )

            # Update blur kernel
            for chan in range(3):
                psf, gradk = update_kernel(gradk, u[..., chan], psf, image[..., chan])

            lambd = lambd / 2

    else:
        # Update sharp image
        with multiprocessing.Pool(processes=3) as pool:
            u = np.dstack(pool.starmap(
                loop_update_image,
                [(image[..., chan], u[..., chan], lambd, psf, iterations) for chan in range(C)]
            )
            )


    return u[pad_v:-pad_v, pad_h:-pad_h, ...], psf

@utils.timeit
def processing_FAST(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(11, 8)

    # Make a non-blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 12, 50, blind=False)

    return pic.astype(np.uint8)

@utils.timeit
def processing_BLIND(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)

    # Generate a dumb blur kernel as point spread function
    psf = np.ones((7, 7))
    psf /= np.sum(psf)

    # Make a blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 0.006, 50, blind=True)

    return pic.astype(np.uint8)


@utils.timeit
def processing_MYOPE(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)

    # Generate a guessed blur kernel as point spread function
    psf = utils.kaiser_kernel(11, 8)

    # Make a blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 1, 50, blind=True)

    return pic.astype(np.uint8)


def save(pic, name):
    with Image.fromarray(pic) as output:
        output.save(join(dest_path, picture + "-" + name + ".jpg"),
                    format="jpeg",
                    optimize=True,
                    progressive=True,
                    quality=90)


if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    images = ["blured.jpg"]

    for picture in images:

        with Image.open(join(source_path, picture)) as pic:

            """
            The "BEST" algorithm resamples the image × 2, applies the deconvolution and
            then sample it back. It's good to dilute the noise on low res images.

            The "FAST" algorithm is a direct method, more suitable for high res images
            that will be resized anyway. It's twice as fast and almost as good.
            """

            pic_fast = processing_FAST(pic)
            save(pic_fast, "fast-v3")

            pic_myope = processing_MYOPE(pic)
            save(pic_myope, "myope-v3")

            pic_blind = processing_BLIND(pic)
            save(pic_blind, "blind-v3")
