# -*- coding: utf-8 -*-
'''
Created on 30 avr. 2017

@author: aurelien

This script shows an implementation of a blind deconvolution
'''

import warnings
from os.path import join, isfile
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import pyfftw
import pickle

from lib import tifffile

warnings.simplefilter("ignore", (DeprecationWarning, UserWarning))

from lib import utils
from lib import deconvolution as dc


def pad_image(image, pad, mode="edge"):
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


def build_pyramid(psf_size, lambd):
    """
    To speed-up the deconvolution, the PSF is estimated successively on smaller images of increasing sizes. This function
    computes the intermediates sizes and regularization factors
    """

    images = [1]
    kernels = [psf_size]
    lambdas = [lambd]


    while kernels[-1] > 3:
        kernels.append(int(np.floor(kernels[-1] / np.sqrt(2))))
        images.append(images[-1] / np.sqrt(2))
        lambdas.append(lambdas[-1] / 2.1)

        if kernels[-1] % 2 == 0:
            kernels[-1] -= 1

        if kernels[-1] < 3:
            kernels[-1] = 3

    return images, kernels, lambdas


from skimage.restoration import denoise_tv_chambolle

@utils.timeit
def deblur_module(pic, filename, dest_path, blur_width, confidence=10, bias=1e-4, step=1e-3, bits=8,
                  iterations=100, sharpness=0, mask=None, display=True, neighbours=8, correlation=False):
    """
    API to call the debluring process

    :param pic: an image memory object, from PIL or tifffile
    :param filename: string, the name of the file to save
    :param dest_path: string, the path where to save the file
    :param blur_width: integer, the diameter of the blur e.g. the size of the PSF
    :param confidence: float, default 1, max 100, set the confidence you have in your sample. For example, on noisy pictures, 
    use 1 to 10. For a clean low-ISO picture, you can go all the way to 100. A low factor will reduce the convergence, a high 
    factor will allow more noise amplification.
    :param bias: float, the blending parameter between sharp and blurred pictures. Ensure    the convergence of the sharp image.
    Usually between 0.0001 and 0.1
    :param step: float, the gradient-descent factor. Normal is 1e-3. Increase it to converge faster, but be careful because
    it could diverge more as well.
    :param bits: integer, default is 8 meaning the input image is encoded with 8 bits/channel. Use 16 if you input 16 bits
    tiff files.
    :param iterations: float, default is 1, meaning that the base number of iterations to perform are the width of the blur.
    While this works in most cases, complicated blurs need extra care. Set it > 1 in conjunction with a smaller step
    when more iterations are needed.
    :param mask: list of 4 integers, the region on which the blur will be estimated to speed-up the process.
    :param neighbours: set the number of pixels used to compute the total variation regularization. 2 is fast but will over-smooth
    and thus reduce the convergence. 4 is good but might blur the edges, 8 is better but slower and might be too permissive in some cases. 
    With 8, you might want to decrease the confidence factor.
    :param sharpness: this applies a final unsharp mask using the input picture as the blurry version and the deconvoluted picture as the sharp one. Put 0
    to disable this feature, and 1 to apply it full throttle.

    :param display: Pop-up a control window at the end of the blur estimation to check the solution before runing it on
    the whole picture
    :return:
    """
    # TODO : refocus http://web.media.mit.edu/~bandy/refocus/PG07refocus.pdf
    # TODO : extract foreground only https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut

    pic = np.ascontiguousarray(pic, dtype=np.float32)
    
    if isfile('fftw_wisdom.pickle'):
        with open('fftw_wisdom.pickle', 'rb') as f:
            pyfftw.import_wisdom(pickle.load(f))
            print("FFT profiles loaded")
    else:
        print("No FFT profiles detected. They will be created at the end of the session")

    # Verifications
    if blur_width < 3:
        raise ValueError("The blur width should be at least 3 pixels.")

    if blur_width % 2 == 0:
        raise ValueError("The blur width should be odd. You can use %i." % (blur_width + 1))
        
    if confidence > 100:
        raise ValueError("The confidence factor is limited to 100. Try increasing the bias instead")

    # Set the bit-depth
    samples = 2**bits - 1

    # Rescale the RGB values between 0 and 1
    pic = pic / samples

    # Make the picture dimensions odd to avoid ringing on the border of even pictures. We just replicate the last row/column
    odd_vert = False
    odd_hor = False

    if pic.shape[0] % 2 == 0:
        pic = pad_image(pic, ((1, 0), (0, 0))).astype(np.float32)
        odd_vert = True
        print("Padded vertically")

    if pic.shape[1] % 2 == 0:
        pic = pad_image(pic, ((0, 0), (1, 0))).astype(np.float32)
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
    
    # Rescale the lambda parameter
    confidence = 1000 * confidence
    
    # Set the error
    epsilon = dc.best_param(pic, confidence, M, N)

    print("\n===== BLIND ESTIMATION OF BLUR =====")

    # Construct the mask for the blur estimation
    if mask:
        # Check the mask size
        if ((mask[1] - mask[0]) % 2 == 0 or (mask[3] - mask[2]) % 2 == 0):
            raise ValueError("The mask dimensions should be odd. You could use at least %i×%i pixels." % (
                blur_width + 2, blur_width + 2))

        u_masked = pic[mask[0]:mask[1], mask[2]:mask[3], ...].copy()
        i_masked = pic[mask[0]:mask[1], mask[2]:mask[3], ...]

    else:
        u_masked = pic.copy()
        i_masked = pic
        

    # Build the intermediate sizes and factors
    images, kernels, lambdas = build_pyramid(MK, confidence)
    k_prec = MK
    for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):
        print("======== Pyramid step %1.3f ========" % i)

        # Resize blured, deblured images and PSF from previous step
        if i != 1:
            im = ndimage.zoom(i_masked, (i, i, 1)).astype(np.float32)
        else:
            im = i_masked

        psf = ndimage.zoom(psf, (k / k_prec, k / k_prec, 1)).astype(np.float32)
        dc.normalize_kernel(psf, k)

        u_masked = ndimage.zoom(u_masked, (im.shape[0] / u_masked.shape[0], im.shape[1] / u_masked.shape[1], 1))

        vert_odd = False
        hor_odd = False

        # Pad to ensure oddity
        if pic.shape[0] % 2 == 0:
            hor_odd = True
            im = pad_image(im, ((1, 0), (0, 0))).astype(np.float32)
            u_masked = pad_image(u_masked, ((1, 0), (0, 0))).astype(np.float32)
            print("Padded vertically")

        if pic.shape[1] % 2 == 0:
            vert_odd = True
            im = pad_image(im, ((0, 0), (1, 0))).astype(np.float32)
            u_masked = pad_image(u_masked, ((0, 0), (1, 0))).astype(np.float32)
            print("Padded horizontally")

        # Pad for FFT
        pad = np.floor(k / 2).astype(int)
        u_masked = pad_image(u_masked, (pad, pad))

        # Make a blind Richardson-Lucy deconvolution on the RGB signal
        dc.richardson_lucy_MM(im, u_masked, psf, bias, im.shape[0], im.shape[1], 3, k, int(iterations / i), step, l, epsilon, neighbours, blind=True, correlation=correlation)

        
        # Unpad FFT because this image is resized/reused the next step
        u_masked = u_masked[pad:-pad, pad:-pad, ...]

        # Unpad oddity for same reasons
        if vert_odd:
            u_masked = u_masked[1:, :, ...]

        if hor_odd:
            u_masked = u_masked[:, 1:, ...]

        k_prec = k

    # Display the control preview
    if display:
        psf_check = (psf - np.amin(psf))
        psf_check = psf_check / np.amax(psf_check)
        plt.imshow(psf_check, interpolation="lanczos", filternorm=1, aspect="equal", vmin=0, vmax=1)
        plt.show()
        plt.imshow(u_masked, interpolation="lanczos", filternorm=1, aspect="equal", vmin=0,
                   vmax=1)
        plt.show()

    print("\n===== REGULAR DECONVOLUTION =====")

    u = pic.copy()

    # Build the intermediate sizes and factors
    k_prec = MK
    for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):
        print("======== Pyramid step %1.3f ========" % i)

        # Resize blured, deblured images and PSF from previous step
        if i != 1:
            im = ndimage.zoom(pic, (i, i, 1)).astype(np.float32)
            psf_loc = ndimage.zoom(psf, (k / k_prec, k / k_prec, 1)).astype(np.float32)
            dc.normalize_kernel(psf_loc, k)
        else:
            im = pic
            psf_loc = psf

        u = ndimage.zoom(u, (im.shape[0] / u.shape[0], im.shape[1] / u.shape[1], 1))

        vert_odd = False
        hor_odd = False

        # Pad to ensure oddity
        if pic.shape[0] % 2 == 0:
            hor_odd = True
            im = pad_image(im, ((1, 0), (0, 0))).astype(np.float32)
            u = pad_image(u, ((1, 0), (0, 0))).astype(np.float32)
            print("Padded vertically")

        if pic.shape[1] % 2 == 0:
            vert_odd = True
            im = pad_image(im, ((0, 0), (1, 0))).astype(np.float32)
            u = pad_image(u, ((0, 0), (1, 0))).astype(np.float32)
            print("Padded horizontally")

        # Pad for FFT
        pad = np.floor(k / 2).astype(int)
        u = pad_image(u, (pad, pad))

        # Make a non-blind Richardson-Lucy deconvolution on the RGB signal
        dc.richardson_lucy_MM(im, u, psf_loc, bias, im.shape[0], im.shape[1], 3, k, int(iterations / i), step, l, epsilon, neighbours, blind=False)

        # Unpad FFT because this image is resized/reused the next step
        u = u[pad:-pad, pad:-pad, ...]

        # Unpad oddity for same reasons
        if vert_odd:
            u = u[1:, :, ...]

        if hor_odd:
            u = u[:, 1:, ...]

    # Unsharp mask to boost a bit the sharpness
    u = (1 + sharpness) * u - sharpness * pic

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
    
    with open('fftw_wisdom.pickle', 'wb') as f:
        pickle.dump(pyfftw.export_wisdom(), f)


if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    # Uncomment the following line if you run into a memory error
    # CPU = 1

    picture = "blured.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [478, 478 + 255, 715, 715 + 255]
        deblur_module(pic, picture + "-v26-2", dest_path, 5, mask=mask, display=True, confidence=80, neighbours=2, bias=5e-4)
        deblur_module(pic, picture + "-v26-4", dest_path, 5, mask=mask, display=True, confidence=100, neighbours=4, bias=5e-4)
        deblur_module(pic, picture + "-v26-8", dest_path, 5, mask=mask, display=True, confidence=100, neighbours=8, bias=5e-4)
        pass

    picture = "IMG_9584-900.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [101, 101 + 255, 67, 67 + 255]
        #deblur_module(pic, picture + "-v26-4", dest_path, 3, display=True, mask=mask, sharpness=0.5, confidence=100, neighbours=8, bias=1e-4)
        #deblur_module(pic, picture + "-v24-8", dest_path, 3, display=False, mask=mask, sharpness=0.5, confidence=10, neighbours=8, bias=5e-4)
        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [1111, 1111 + 513, 3383, 3383 + 513]
        #deblur_module(pic, picture + "-v22-2", dest_path, 11, mask=mask, display=True, iterations=100, confidence=10, neighbours=8)
        pass

    picture = "P1030302.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [1492, 1492 + 555, 476, 476 + 555]
        #deblur_module(pic, picture + "-v26", dest_path, 37, mask=mask, display=True, iterations=100, sharpness=0, confidence=50, neighbours=4, bias=0)
        pass

    picture = "153412.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [1484, 1484 + 501, 3428, 3428 + 501]
        #deblur_module(pic, picture + "-v24", dest_path, 5, mask=mask, display=False, confidence=5, neighbo	urs=8, bias=1e-6)
        pass

    # TIFF input example
    source_path = "/home/aurelien/Exports/2017-11-19-Shoot Fanny Wong/export"
    picture = "Shoot Fanny Wong-0146-_DSC0426--PHOTOSHOP.tif"
    #pic = tifffile.imread(join(source_path, picture))
    mask = [1914, 1914 + 171, 718, 1484 + 171]
    #deblur_module(pic, picture + "-blind-v18", dest_path, 5, 1000, 3.0, 1e-3, mask=mask, bits=16)

