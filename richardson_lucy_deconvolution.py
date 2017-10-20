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
import os
from os.path import join

import numpy as np
import scipy.signal
from PIL import Image, ImageDraw
from numba import float32, jit, int16
from scipy import misc

from lib import utils

CPU = os.cpu_count()

try:
    from pyculib.fft import FFTPlan, fft_inplace, ifft_inplace


    # @jit(cache=True)
    def fftconvolve(im, psf, domain):
        """"GPU FFT convolution via CUDA/OpenCL

        """
        Nx1, Nx2 = im.shape
        NKx1, NKx2, = psf.shape
        dtype = np.complex64

        # Define the best square fit
        # square = np.maximum(Nx1, Nx2) + NKx1 -1
        # im = np.pad(im, ((0, square - Nx1), (0, square - Nx2)), mode="constant", constant_values=0)
        # psf = np.pad(psf, ((0, square - NKx1), (0, square - NKx2)), mode="constant", constant_values=0)

        im = np.pad(im, ((0, NKx1 - 1), (0, NKx2 - 1)), mode="constant", constant_values=0)
        psf = np.pad(psf, ((0, Nx1 - 1), (0, Nx2 - 1)), mode="constant", constant_values=0)
        size = im.shape[0] * im.shape[1]

        fft_inplace(im)
        im = np.fft.fftshift(im)
        fft_inplace(psf)
        psf = np.fft.fftshift(psf)

        result = ifft_inplace(np.fft.fftshift(im * psf)).real.astype(np.float32)

        if domain == "same":
            return result[np.floor(NKx1 / 2).astype(int):-np.floor(NKx1 / 2).astype(int) - 1,
                   np.floor(NKx2 / 2).astype(int):-np.floor(NKx2 / 2).astype(int) - 1]
        elif domain == "valid":
            return result[NKx1 - 1:-NKx1 + 1, NKx2 - 1:-NKx2 + 1]
        elif domain == "full":
            return result
        else:
            raise ValueError

except:
    print("No Cuda support, fallback on CPU")
    fftconvolve = scipy.signal.fftconvolve


@jit
def find_nearest_power(x):
    return 1 << (x - 1).bit_length()

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
    return np.ascontiguousarray(grad, np.float32)


@jit(cache=True)
def pad_image(image: np.ndarray, pad: tuple, mode="edge"):
    R = np.pad(image[..., 0], pad, mode=mode)
    G = np.pad(image[..., 1], pad, mode=mode)
    B = np.pad(image[..., 2], pad, mode=mode)
    u = np.dstack((R, G, B))
    return np.ascontiguousarray(u, np.ndarray)


@jit(cache=True)
def unpad_image(image: np.ndarray, pad: tuple):
    return np.ascontiguousarray(image[pad[0]:-pad[0], pad[1]:-pad[1], ...], np.ndarray)

@jit(float32[:, :](float32[:, :], float32[:, :]), cache=True)
def trim_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Trim lines and columns out of the mask

    :param image: masked image with non-zeros values inside the mask and 0 outside
    :return:
    """

    image = image[mask[0]:mask[1], mask[2]:mask[3]]
    return image


@jit(float32[:, :](float32[:, :], float32[:, :], float32[:, :]), cache=True)
def convolve_kernel(u, psf, image):
    return fftconvolve(np.rot90(u, 2), fftconvolve(u, psf, "valid") - image, "valid").astype(np.float32)


@jit(float32[:, :](float32[:, :], float32[:, :], float32[:, :]), cache=True)
def convolve_image(u, psf, image):
    return fftconvolve(fftconvolve(u, psf, "valid") - image, np.rot90(psf), "full").astype(np.float32)


@jit(float32[:, :](float32[:, :], float32[:, :], float32, float32[:, :]), cache=True)
def update_image(image, u, lambd, psf):
    """Update one channel only (R, G or B)

    :param image:
    :param u:
    :param lambd:
    :param psf:
    :return:
    """

    # Total Variation Regularization
    gradUdata = convolve_image(u, psf, image)
    gradu = gradUdata - lambd * divTV(u)[:gradUdata.shape[0], :gradUdata.shape[1]]
    u = update_values(u[:gradUdata.shape[0], :gradUdata.shape[1]],
                      weight_update(5e-3, u[:gradUdata.shape[0], :gradUdata.shape[1]], gradu), gradu)

    # Normalize for 8 bits RGB values
    u = np.clip(u, 0.0000001, 255)

    return np.ascontiguousarray(u, np.float32)


@jit(float32[:, :](float32[:, :], float32[:, :], int16, float32[:, :], int16), cache=True)
def loop_update_image(image, u, lambd, psf, iterations):
    for i in range(iterations):
        print(" = Iteration :", i, " =")
        u = update_image(image, u, lambd, psf)
        lambd = 0.99 * lambd

    return np.ascontiguousarray(u, np.float32)


@jit(float32[:, :](int16, float32, float32, float32), cache=True)
def build_pyramid(psf_size: int, lambd: float, scaling: float = 1.9, max_lambd: float = 1) -> list:
    # Initialize the pyramid of variables
    lambdas = [lambd]
    images = [1]
    kernels = [psf_size]

    while (lambdas[-1] * scaling < max_lambd and kernels[-1] > 3):
        lambdas.append(lambdas[-1] * scaling)
        kernels.append(kernels[-1] - 2)
        images.append(images[-1] / scaling)

    return images, kernels, lambdas


@jit(float32[:, :](float32[:, :]), cache=True)
def normalize_kernel(kern: np.ndarray) -> np.ndarray:
    kern[kern < 0] = 0
    return (kern / np.sum(kern)).astype(np.float32)


@jit(float32[:, :](float32, float32[:, :], float32[:, :]), cache=True)
def weight_update(factor: float, array: np.ndarray, grad_array: np.ndarray) -> np.ndarray:
    return (factor * np.amax(array) / np.maximum(1e-31, np.amax(np.abs(grad_array)))).astype(np.float32)


@jit(float32[:, :](float32[:, :], float32, float32[:, :]), cache=True)
def update_values(target: np.ndarray, factor: float, source: np.ndarray) -> np.ndarray:
    return (target - factor * source).astype(np.float32)


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
    MK, NK = psf.shape
    M, N, C = image.shape
    pad = np.floor(MK / 2).astype(int)
    pad = (pad, pad)

    u = pad_image(u, pad).astype(np.float32)

    print("working on image :", u.shape)

    if blind:
        # Blind or myopic deconvolution

        if mask != None:
            masked_image = trim_mask(image, mask)

        for i in range(iterations):
            print(" = Iteration :", i, " =")

            with multiprocessing.Pool(processes=CPU) as pool:
                u = np.dstack(pool.starmap(update_image,
                                           [(image[..., chan], u[..., chan], lambd, psf) for chan in range(C)])).astype(
                    np.float32)

            lambd = 0.99 * lambd

            # Extract the portion of the source image and the deconvolved image under the mask
            if mask != None:
                masked_u = pad_image(trim_mask(unpad_image(u, pad), mask), pad).astype(np.float32)

            # Update the blur kernel
            grad_psf = np.zeros_like(psf).astype(np.float32)

            with multiprocessing.Pool(processes=CPU) as pool:
                if mask != None:
                    grad_psf = grad_psf + pool.starmap(convolve_kernel,
                                                       [(masked_u[..., chan], psf, masked_image[..., chan]) for chan in
                                                        range(C)])[0]
                else:
                    grad_psf = grad_psf + pool.starmap(convolve_kernel,
                                                       [(u[..., chan], psf, image[..., chan]) for chan in range(C)])[0]

            psf = normalize_kernel(update_values(psf, weight_update(1e-3, psf, grad_psf), grad_psf))

    else:
        # Regular non-blind RL deconvolution

        # Update sharp image
        with multiprocessing.Pool(processes=CPU) as pool:
            u = np.dstack(pool.starmap(loop_update_image,
                                       [(image[..., chan], u[..., chan], lambd, psf, iterations) for chan in range(C)]))

    u = unpad_image(u, pad)
    return u.astype(np.float32), psf

@utils.timeit
@jit(cache=True)
def deblur_module(pic: np.ndarray, filename: str, blur_type: str, quality: int, artifacts_damping: float,
                  deblur_strength: float, blur_width: int = 3,
                  blur_strength: int = 1, refine: bool = False, mask: np.ndarray = None, backvsmask_ratio: float = 0,
                  debug: bool = False, psf: np.ndarray = None):
    """This mimics a Darktable module inputs where the technical specs are hidden and translated into human-readable parameters.

    It's an interface between the regular user and the geeky actual deconvolution parameters

    :param pic: Blured image in RGB 8 bits
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
    # ! TODO : refocus

    pic = np.ascontiguousarray(pic, np.float32)

    if blur_type == "auto":
        print("\n===== KERNEL INIT =====")

        images, kernels, lambdas = build_pyramid(blur_width, artifacts_damping)

        u = pic.copy(order="C")
        psf = utils.uniform_kernel(kernels[-1]).astype(np.float32)
        runs = 1
        iterations = np.maximum(np.ceil(quality / len(kernels)), kernels[-1]).astype(int)

        for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):

            print("==== Pyramid level", runs, " ====")

            # Scale the previous deblured image to the dimensions of i
            new_size = (round(i * pic.shape[0]) + k - 1, round(i * pic.shape[1]) + k - 1)
            if i != 1:
                # For some reasons, rescaling by 1 gives weird sizes
                im = np.ascontiguousarray(misc.imresize(pic, new_size, interp="lanczos", mode="RGB"), np.float32)
            else:
                im = pic.copy(order="C")

            u = np.ascontiguousarray(misc.imresize(u, im.shape, interp="lanczos"), np.float32)

            # Scale the PSF
            if k != kernels[-1]:
                psf = normalize_kernel(misc.imresize(psf, (k, k), interp="lanczos"))

            # Make a blind Richardson- Lucy deconvolution on the RGB signal
            u, psf = richardson_lucy(im, u, psf, l, iterations, blind=True)

            runs = runs + 1

        print("-> Kernel init done")
    else:
        u = pic.copy()

    if blur_type == "gaussian":
        psf = utils.gaussian_kernel(blur_strength, blur_width)

    if blur_type == "kaiser":
        psf = utils.kaiser_kernel(blur_strength, blur_width)

    if refine:
        if mask:
            # Masked blind or myopic deconvolution
            print("\n===== BLIND MASKED REFINEMENT =====")
            iter_background = np.round(quality * backvsmask_ratio).astype(int)
            iter_mask = np.round(quality - iter_background).astype(int)

            u, psf = richardson_lucy(pic, u, psf, artifacts_damping, iter_mask, blind=True, mask=mask)
            print("-> Masked blind deconv done")

            if iter_background != 0:
                print("\n===== BLIND UNMASKED REFINEMENT =====")
                u, psf = richardson_lucy(pic, u, psf, artifacts_damping, iter_background, blind=True)
                print("-> Unmasked blind deconv done")
        else:
            # Unmasked blind or myopic deconvolution
            print("\n===== BLIND UNMASKED REFINEMENT =====")
            u, psf = richardson_lucy(pic, u, psf, artifacts_damping, quality, blind=True)
            print(" -> Unmasked blind deconv done")

        if deblur_strength > 0:
            # Apply a last round of non-blind optimization to enhance the sharpness
            print("\n===== REGULAR DECONVOLUTION =====")
            u, psf = richardson_lucy(pic, u, psf, artifacts_damping, deblur_strength)
            print("-> Regular deconv done")

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


def interpolated(pic, filename):
    u_top, psf = deblur_module(pic, "blind-v6-top", "auto", 1000, 0.1, 0, mask=[150, 150 + 256, 600, 600 + 256],
                               refine=True,
                               backvsmask_ratio=0.1, blur_width=7, debug=True)

    grad_top = divTV(u_top)
    weight_top = 1 / weight_update(3, u_top, grad_top)

    u_middle, psf = deblur_module(pic, "blind-v6-middle", "auto", 1000, 0.005, 0, mask=[150, 150 + 256, 600, 600 + 256],
                                  refine=True,
                                  backvsmask_ratio=0.1, debug=True, blur_width=9)

    grad_middle = divTV(u_middle)
    weight_middle = 1 / weight_update(3, u_middle, grad_middle)

    u_bottom, psf = deblur_module(pic, "blind-v6-bottom", "auto", 1000, 0.005, 2, mask=[150, 150 + 256, 600, 600 + 256],
                                  refine=True,
                                  backvsmask_ratio=0.1, debug=True, blur_width=11)

    grad_bottom = divTV(u_bottom)
    weight_bottom = 1 / weight_update(3, u_bottom, grad_bottom)

    u = (weight_top * u_top + weight_middle * u_middle + weight_bottom * u_bottom) / (
    weight_top + weight_middle + weight_top)

    u = np.clip(u, 0.00001, 255)

    save(u, "interpolated")


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

    images = ["blured.jpg"]

    for picture in images:
        with Image.open(join(source_path, picture)) as pic:
            """
            The "BEST" algorithm resamples the image × 2, applies the deconvolution and
            then sample it back. It's good to dilute the noise on low res images.

            The "FAST" algorithm is a direct method, more suitable for high res images
            that will be resized anyway. It's twice as fast and almost as good.
            """

            # pic_fast = processing_FAST(pic)
            #deblur_module(pic, "fast-v4", "kaiser", 0, 0.05, 50, blur_width=11, blur_strength=8)

            # pic_myope = processing_MYOPE(pic)
            #deblur_module(pic, "myope-v4", "kaiser", 10, 0.05, 50, blur_width=11, blur_strength=8, mask=[150, 150 + 256, 600, 600 + 256], refine=True,)

            # pic_blind = processing_BLIND(pic)
            deblur_module(pic, "blind-v6-middle", "auto", 40, 0.005, 10, mask=[150, 150 + 256, 600, 600 + 256],
                          refine=True,
                          backvsmask_ratio=0.2, debug=True, blur_width=9)

            interpolated(pic, "interpolated")


    with Image.open(join(source_path, "DSC_1168_test.jpg")) as pic:

        # deblur_module(pic, "test-v6", "auto", 5, 0.005, 5, mask=[1271, 1271 + 256, 3466, 3466 + 256], refine=False, backvsmask_ratio=0, blur_width=15)
        pass
