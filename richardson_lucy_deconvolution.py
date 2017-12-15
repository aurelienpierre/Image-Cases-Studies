# -*- coding: utf-8 -*-
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

import warnings
from os.path import join
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

from lib import tifffile

warnings.simplefilter("ignore", (DeprecationWarning, UserWarning))

from lib import utils
from lib import deconvolution as dc


def build_pyramid(psf_size, lambd):
    """
    To speed-up the deconvolution, the PSF is estimated successively on smaller images of increasing sizes. This function
    computes the intermediates sizes and regularization factors
    """

    lambdas = [lambd]
    images = [1]
    kernels = [psf_size]

    image_multiplier = np.sqrt(2)
    lambda_multiplier = 2.1

    while kernels[-1] > 3:
        lambdas.append(lambdas[-1] * lambda_multiplier)
        kernels.append(int(np.floor(kernels[-1] / image_multiplier)))
        images.append(images[-1] / image_multiplier)

        if kernels[-1] % 2 == 0:
            kernels[-1] -= 1

        if kernels[-1] < 3:
            kernels[-1] = 3

    return images, kernels, lambdas

@utils.timeit
def deblur_module(pic, filename, dest_path, blur_width, lambd, tau, step_factor, bits=8, mask=None):
    """
    This mimics a Darktable module inputs where the technical specs are hidden and translated into human-readable parameters.

    It's an interface between the regular user and the geeky actual deconvolution parameters

    :return:
    """
    # TODO : refocus http://web.media.mit.edu/~bandy/refocus/PG07refocus.pdf
    # TODO : extract foreground only https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut

    pic = np.ascontiguousarray(pic, dtype=np.float32)

    # Verifications
    if blur_width < 3:
        raise ValueError("The blur width should be at least 3 pixels.")

    if blur_width % 2 == 0:
        raise ValueError("The blur width should be odd. You can use %i." % (blur_width + 1))

    # Set the bit-depth
    samples = 2**bits - 1

    # Rescale the RGB values between 0 and 1
    pic = pic / samples

    # Make the picture dimensions odd to avoid ringing on the border of even pictures. We just replicate the last row/column
    odd_vert = False
    odd_hor = False

    if pic.shape[0] % 2 == 0:
        pic = dc.pad_image(pic, ((1, 0), (0, 0))).astype(np.float32)
        odd_vert = True
        print("Padded vertically")

    if pic.shape[1] % 2 == 0:
        pic = dc.pad_image(pic, ((0, 0), (1, 0))).astype(np.float32)
        odd_hor = True
        print("Padded horizontally")

    # Construct a blank PSF
    psf = utils.uniform_kernel(blur_width)
    psf = np.dstack((psf, psf, psf))

    # Get the dimensions once for all
    MK = blur_width
    M = pic.shape[0]
    N = pic.shape[1]
    C = pic.shape[2]

    print("\n===== BLIND ESTIMATION OF BLUR =====")

    # Construct the mask for the blur estimation
    if mask:
        # Check the mask size
        if ((mask[1] - mask[0]) * (mask[3] - mask[2])) < blur_width ** 4:
            raise ValueError("The mask is too small regarding the blur width. It should be at least %i×%i pixels." % (
            blur_width ** 2 + 2, blur_width ** 2 + 2))

        if ((mask[1] - mask[0]) % 2 == 0 or (mask[3] - mask[2]) % 2 == 0):
            raise ValueError("The mask dimensions should be odd. You could use at least %i×%i pixels." % (
            blur_width ** 2 + 2, blur_width ** 2 + 2))

        u_masked = pic[mask[0]:mask[1], mask[2]:mask[3], ...].copy()
        i_masked = pic[mask[0]:mask[1], mask[2]:mask[3], ...]
    else:
        u_masked = pic.copy()
        i_masked = pic

    # Build the intermediate sizes and factors
    images, kernels, lambdas = build_pyramid(MK, lambd)

    k_prec = MK

    for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):
        print("== Pyramid step", i, "==")

        # Resize blured, deblured images and PSF from previous step
        if i != 1:
            im = ndimage.zoom(i_masked, (i, i, 1)).astype(np.float32)
        else:
            im = i_masked

        psf = ndimage.zoom(psf, (k / k_prec, k / k_prec, 1)).astype(np.float32)
        dc.normalize_kernel(psf, k)

        u_masked = ndimage.zoom(u_masked, (im.shape[0] / u_masked.shape[0], im.shape[1] / u_masked.shape[1], 1))

        # Pad to ensure oddity
        if pic.shape[0] % 2 == 0:
            im = dc.pad_image(im, ((1, 0), (0, 0))).astype(np.float32)
            u_masked = dc.pad_image(u_masked, ((1, 0), (0, 0))).astype(np.float32)
            print("Padded vertically")

        if pic.shape[1] % 2 == 0:
            im = dc.pad_image(im, ((0, 0), (1, 0))).astype(np.float32)
            u_masked = dc.pad_image(im, ((0, 0), (1, 0))).astype(np.float32)
            print("Padded horizontally")

        # Pad for FFT
        pad = np.floor(k / 2).astype(int)
        u_masked = dc.pad_image(u_masked, (pad, pad))

        # Make a blind Richardson-Lucy deconvolution on the RGB signal
        print(im.shape)
        dc.richardson_lucy_MM(im, u_masked, psf, l, tau, step_factor, im.shape[0], im.shape[1], 3, k)

        k_prec = k

    # Display the control preview
    psf_check = (psf - np.amin(psf))
    psf_check = psf_check / np.amax(psf_check)
    plt.imshow(psf_check, interpolation="lanczos", filternorm=1, aspect="equal", vmin=0, vmax=1)
    plt.show()
    plt.imshow(u_masked[pad:-pad, pad:-pad, ...], interpolation="lanczos", filternorm=1, aspect="equal")
    plt.show()

    print("\n===== REGULAR DECONVOLUTION =====")

    # Pad every edge of the image to avoid boundaries problems during convolution
    pad = np.floor(blur_width / 2).astype(int)
    u = dc.pad_image(pic, (pad, pad))
    dc.richardson_lucy_MM(pic, u, psf, lambd, tau, step_factor, M, N, C, MK, blind=False)

    # Remove the FFT padding
    u = u[pad:-pad, pad:-pad, ...]

    # if the picture has been padded to make it odd, unpad it to get the original size
    if odd_hor:
        u = u[:, 1:, ...]
    if odd_vert:
        u = u[1:, :, ...]

    # Clip extreme values
    np.clip(u, 0, 1, out=u)

    # Convert to 16 bits RGB
    u = u * (2 ** 16 - 1)

    utils.save(u, filename, dest_path)


if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    # Uncomment the following line if you run into a memory error
    # CPU = 1

    picture = "blured.jpg"
    with Image.open(join(source_path, picture)) as pic:
        deblur_module(pic, picture + "-blind-v18", dest_path, 5, 5000, 3.0, 3e-3, mask=[478, 478 + 255, 715, 715 + 255])

        pass

    picture = "Shoot-Sienna-Hayes-0042-_DSC0284-sans-PHOTOSHOP-WEB.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # deblur_module(pic, picture + "blind-v18", dest_path, 13, 1000, 3.0, 1e-3, mask=[661, 661 + 255, 532, 532 + 255])

        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:
        #deblur_module(pic, picture + "blind-v18", dest_path, 9, 5000, 0.01, 3e-3, mask=[631, 631+255, 2826, 2826+255])

        pass

    picture = "IMG_9584-900.jpg"
    with Image.open(join(source_path, picture)) as pic:
        #deblur_module(pic, picture + "test-v18", dest_path, 5, 30000, 1.0, 3e-3, mask=[101, 101+257, 67, 67+257])
        pass

    picture = "P1030302.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # deblur_module(pic, picture + "-blind-v18-best", dest_path, 29, 8000, 2., 3e-3, mask=[3560, 3560+512, 16, 16+512])

        pass

    # JPEG input example
    picture = "153412.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [1484, 1484 + 255, 3228, 3228 + 255]
        deblur_module(pic, picture + "-blind-v18-best", dest_path, 7, 30000, 2., 1e-3, mask=mask)

        pass

    # TIFF input example
    source_path = "/home/aurelien/Exports/2017-11-19-Shoot Fanny Wong/export"
    picture = "Shoot Fanny Wong-0146-_DSC0426--PHOTOSHOP.tif"
    pic = tifffile.imread(join(source_path, picture))

    mask = [1914, 1914 + 171, 718, 1484 + 171]
    #deblur_module(pic, picture + "-blind-v18", dest_path, 5, 1000, 3.0, 1e-3, mask=mask, bits=16)
