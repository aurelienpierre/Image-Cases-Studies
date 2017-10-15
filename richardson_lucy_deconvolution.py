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
from PIL import Image, ImageDraw
from numba import float32, int16, jit
from scipy import misc
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


# @jit(float32[:, :](float32[:, :], float32[:, :], int16, float32[:, :]), cache=True)
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

        lambd = lambd * 0.99

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
    # Compute the new PSF
    gradk = gradk + fftconvolve(np.rot90(u, 2), fftconvolve(u, psf, "valid") - image, "valid")
    sh = 1e-3 * np.amax(psf) / np.maximum(1e-31, np.amax(np.abs(gradk)))
    psf = psf - sh * gradk

    # Normalize the kernel
    psf[psf < 0] = 0
    psf = psf / np.sum(psf)

    return psf, gradk


@jit(float32[:, :](float32[:, :], float32[:, :]), cache=True)
def trim_mask(image: np.ndarray, mask: np.ndarray):
    """Trim lines and columns out of the mask

    :param image: masked image with non-zeros values inside the mask and 0 outside
    :return:
    """

    image = image[mask[0]:mask[1], mask[2]:mask[3]]
    return image


# @jit(float32[:,:](float32[:,:], float32[:,:], float32, int16, bool[:,:]), cache=True)
@utils.timeit
def richardson_lucy(image: np.ndarray, u: np.ndarray, psf: np.ndarray, lambd: float, iterations: int,
                    blind: bool = False, mask: np.ndarray = None) -> np.ndarray:
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
    u = pad_image(u, pad_v, pad_h)

    if blind:
        # Blind or myopic deconvolution
        gradk = np.zeros((MK, NK))

        if mask != None:
            masked_image = trim_mask(image, mask)

        for i in range(iterations):
            print("Iteration :", i)
            # Update sharp image
            with multiprocessing.Pool(processes=3) as pool:
                u = np.dstack(pool.starmap(
                    update_image,
                    [(image[..., chan], u[..., chan], lambd, psf) for chan in range(C)]
                )
                )

            # Extract the portion of the source image and the deconvolved image under the mask
            if mask != None:
                masked_u = pad_image(trim_mask(u, mask), pad_v, pad_h)

            # Update the blur kernel
            for chan in range(3):
                if mask != None:
                    psf, gradk = update_kernel(gradk, masked_u[..., chan], psf, masked_image[..., chan])
                else:
                    psf, gradk = update_kernel(gradk, u[..., chan], psf, image[..., chan])

            lambd = lambd * 0.99


    else:
        # Regular non-blind RL deconvolution

        # Update sharp image
        with multiprocessing.Pool(processes=3) as pool:
            u = np.dstack(pool.starmap(
                loop_update_image,
                [(image[..., chan], u[..., chan], lambd, psf, iterations) for chan in range(C)]
            )
            )


    return u[pad_v:-pad_v, pad_h:-pad_h, ...], psf


@jit(float32[:, :](float32[:, :], int16, float32, float32, float32), cache=True)
def build_pyramid(image, psf_size, lambd, scaling=1.9, max_lambd=0.5):
    # Initialize the pyramid of variables
    lambdas = [lambd]
    images = [image]
    kernels = [psf_size]

    while (lambdas[-1] * scaling < max_lambd and kernels[-1] > 3):
        lambdas.append(lambdas[-1] * scaling)
        kernels.append(kernels[-1] - 2)
        images.append(misc.imresize(image, 1 / scaling, interp="lanczos", mode="RGB").astype(float))

    return images, kernels, lambdas


@jit(float32[:, :](int16), cache=True)
def uniform_kernel(size):
    kern = np.ones((size, size))
    kern /= np.sum(kern)
    return kern

@utils.timeit
def deblur_module(image: np.ndarray, filename: str, blur_type: str, quality: int, artifacts_damping: float,
                  deblur_strength: float, blur_width: int = 3,
                  blur_strength: int = 1, refine: bool = False, mask: np.ndarray = None, backvsmask_ratio: float = 0,
                  debug: bool = False):
    """This mimics a Darktable module inputs where the technical specs are hidden and translated into human-readable parameters

    :param image: Blured image in RGB 8 bits
    :param filename: File name to save the sharp picture
    :param blur_type: kind of blur or "auto" to perform a blind deconvolution. Use "auto" for motion blur or composite blur
    :param blur_width: the width of the blur in px
    :param blur_strength: the strength of the blur, thus the standard deviation of the blur kernel
    :param quality: the quality of the refining (ie the total number of iterations to compute). (10 to 100)
    :param artifacts_damping: the noise and artifacts reduction factor lambda. Typically between 0.00003 and 1. Increase it if smudges, noise, or ringing appear
    :param deblur_strength: the number of debluring iterations to perform,
    :param refine: True or False, decide if the blur kernel should be refined through myopic deconvolution
    :param mask: the coordinates of the rectangular mask to apply on the image to refine the blur kernel from the top-left corner of the image
        in list [y_top, y_bottom, x_left, x_right]
    :param backvsmask_ratio: when a mask is used, the ratio  of weights of the whole image / the masked zone.
        0 means only the masked zone is used, 1 means the masked zone is ignored and only the whole image is taken. 0 is
        runs faster, 1 runs much slower.
    :return:
    """

    pic = np.array(image).astype(np.float32)

    if blur_type == "auto":

        images, kernels, lambdas = build_pyramid(image, blur_width, artifacts_damping)

        u = pic.copy()
        psf = uniform_kernel(kernels[-1])
        runs = 1

        for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):

            print(runs)

            i = np.array(i).astype(np.float32)

            # Scale the previous deblured image
            u = misc.imresize(u, (i.shape[0], i.shape[1]), interp="lanczos")

            # Scale the PSF
            if k != psf.shape[0]:
                psf = misc.imresize(psf, (k, k), interp="lanczos")
                psf[psf < 0] = 0
                psf = psf / np.sum(psf)

            print(psf)
            print(i.shape)

            # Make a blind Richardson- Lucy deconvolution on the RGB signal
            u, psf = richardson_lucy(i, u, psf, l, np.round(quality / 2).astype(int), blind=True)

            runs = runs + 1

        print("Kernel init done")
    else:
        u = pic.copy()

    if blur_type == "gaussian":
        psf = utils.gaussian_kernel(blur_strength, blur_width)

    if blur_type == "kaiser":
        psf = utils.kaiser_kernel(blur_strength, blur_width)

    if refine:
        if mask:
            # Masked blind or myopic deconvolution
            iter_background = np.round(quality * backvsmask_ratio).astype(int)
            iter_mask = np.round(quality - iter_background).astype(int)

            u, psf = richardson_lucy(pic, u, psf, artifacts_damping, iter_mask, blind=True, mask=mask)
            print("Masked blind deconv done")

            if iter_background != 0:
                u, psf = richardson_lucy(pic, u, psf, artifacts_damping, iter_background, blind=True)
                print("Unmasked blind deconv done")
        else:
            # Unmasked blind or myopic deconvolution
            u, psf = richardson_lucy(pic, u, psf, artifacts_damping, quality, blind=True)
            print("Unmasked blind deconv done")

        if deblur_strength > 0:
            # Apply a last round of non-blind optimization to enhance the sharpness
            u, psf = richardson_lucy(pic, u, psf, artifacts_damping, deblur_strength)
            print("Additional deconv done")

    else:
        # Regular Richardson-Lucy deconvolution
        u, psf = richardson_lucy(pic, u, psf, artifacts_damping, deblur_strength)

    if debug:
        # Print the mask in debug mode
        save(u, filename, mask)
    else:
        save(u, filename)

    return u, psf


@utils.timeit
@jit(float32[:, :](float32[:, :]), cache=True)
def processing_FAST(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(11, 8)

    # Make a non-blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 0.05, 50, blind=False)

    save(pic, "fast-v3")

    return pic.astype(np.uint8)

@utils.timeit
@jit(float32[:, :](float32[:, :]), cache=True)
def processing_BLIND(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)
    pic_copy = pic.copy()

    # Generate a dumb blur kernel as point spread function
    size = 3
    psf = np.ones((size, size))
    psf /= np.sum(psf)

    # Draw a 255×255 pixels mask beginning at the coordinate [252, 680] (top-left corner)
    mask = [150, 150 + 256, 600, 600 + 256]

    # Initialize the blur kernel
    for i in [16, 8, 4, 2, 1]:
        ratio = (size + 2) / size

        # Downscale the picture
        if i != 1:
            pic_copy = misc.imresize(pic, 1.0 / i, interp="lanczos", mode="RGB").astype(float)
        else:
            pic_copy = pic

        # Make a blind Richardson- Lucy deconvolution on the RGB signal
        pic_copy, psf = richardson_lucy(pic_copy, psf, 0.005 * i, 5 * i, blind=True)

        # Upscale the PSF
        if i != 1:
            psf = misc.imresize(psf, ratio, interp="lanczos")
            size += 2

    pic, psf = richardson_lucy(pic_copy, psf, 0.005, 20, blind=True, mask=mask)
    pic, psf = richardson_lucy(pic_copy, psf, 0.0005, 10, blind=True)
    pic, psf = richardson_lucy(pic, psf, 0.5, 10)

    # Draw the mask
    save(pic, "blind-v5", mask)

    return pic.astype(np.uint8)


@utils.timeit
@jit(float32[:, :](float32[:, :]), cache=True)
def processing_MYOPE(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)

    # Generate a guessed blur kernel as point spread function
    psf = utils.kaiser_kernel(11, 8)

    # Draw a 255×255 pixels mask beginning at the coordinate [252, 680] (top-left corner)
    mask = [150, 150 + 256, 600, 600 + 256]

    # Make a blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 1, 46, blind=True, mask=mask)
    pic, psf = richardson_lucy(pic, psf, 0.005, 4, blind=True)
    pic, psf = richardson_lucy(pic, psf, 0.5, 5)

    # Draw the mask
    save(pic, "myope-v5", mask)

    return pic.astype(np.uint8)


def save(pic, name, mask=False):
    with Image.fromarray(pic.astype(np.uint8)) as output:
        if mask:
            draw = ImageDraw.Draw(output)
            draw.rectangle([(mask[2], mask[0]), (mask[3], mask[1])], fill=None, outline=128)

        output.save(join(dest_path, picture + "-" + name + ".jpg"),
                    format="jpeg",
                    optimize=True,
                    progressive=True,
                    quality=90)


if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    images = ["DSC_1168_test.jpg"]

    for picture in images:

        with Image.open(join(source_path, picture)) as pic:

            """
            The "BEST" algorithm resamples the image × 2, applies the deconvolution and
            then sample it back. It's good to dilute the noise on low res images.

            The "FAST" algorithm is a direct method, more suitable for high res images
            that will be resized anyway. It's twice as fast and almost as good.
            """

            # pic_fast = processing_FAST(pic)

            # pic_myope = processing_MYOPE(pic)

            # pic_blind = processing_BLIND(pic)

            # deblur_module(pic, "blind-v6", "auto", 5, 0.005, 0, mask=[150, 150 + 256, 600, 600 + 256], refine=True, backvsmask_ratio=0, blur_width=5)

            deblur_module(pic, "test-v6", "auto", 50, 0.005, 10, mask=[1271, 1271 + 256, 3466, 3466 + 256], refine=True,
                          backvsmask_ratio=0, blur_width=15)
