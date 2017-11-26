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

import multiprocessing
import warnings
from os.path import join

import numpy as np

import scipy
from PIL import Image
from numba import float32, jit, int16, boolean
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle
from lib import tifffile

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack

except:
    pass

warnings.simplefilter("ignore", (DeprecationWarning, UserWarning))

from lib import utils
from lib import deconvolution as dc

CPU = multiprocessing.cpu_count()

@jit(cache=True)
def build_pyramid(psf_size, lambd, method):
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

    lambda_max = 0.5
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

    print(images, kernels, lambdas)
    return images, kernels, lambdas

@jit(cache=True)
def process_pyramid(pic, u, psf, lambd, method, epsilon, quality=1):
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

    # Prepare the biggest version
    u = ndimage.zoom(u, (images[-1], images[-1], 1)).astype(np.float32)
    k_prec = Mk
    iterations = int(quality * 10)

    for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)):
        print("== Pyramid step", i, "==")
        odd_vert = False
        odd_hor = False

        # Resize blured, deblured images and PSF from previous step
        if i != 1:
            im = ndimage.zoom(pic, (i, i, 1)).astype(np.float32)

            # Make the picture dimensions odd to avoid ringing on the border of even pictures. We just replicate the last row/column
            if pic.shape[0] % 2 == 0:
                pic = dc.pad_image(pic, ((1, 0), (0, 0))).astype(np.float32)
                odd_vert = True

            if pic.shape[1] % 2 == 0:
                pic = dc.pad_image(pic, ((0, 0), (1, 0))).astype(np.float32)
                odd_hor = True
        else:
            im = pic.copy()

        psf = ndimage.zoom(psf, (k / k_prec, k / k_prec, 1)).astype(np.float32)
        psf = dc._normalize_kernel(psf).astype(np.float32)
        u = ndimage.zoom(u, (im.shape[0] / u.shape[0], im.shape[1] / u.shape[1], 1)).astype(np.float32)

        # Make a blind Richardson-Lucy deconvolution on the RGB signal
        u, psf = method(im, u, psf, l, int(iterations/i), epsilon)

        # if the picture has been padded to make it odd, unpad it to get the original size
        if odd_hor:
            u = u[:, 1:, ...]

        if odd_vert:
            u = u[1:, :, ...]

        k_prec = k

    return u, psf

@jit(cache=True)
def make_preview(image, psf, ratio, mask=None):
    """
    Resize the image, the PSF and the mask to preview the settings on a smaller picture to speed-up the tweaking

    :param image:
    :param psf:
    :param ratio:
    :param mask:
    :return:
    """
    image = ndimage.zoom(image, (ratio, ratio, 1)).astype(np.float32)

    MK_source = psf.shape[0]
    MK = int(MK_source * ratio)

    if MK % 2 == 0:
        MK += 1

    if MK < 3:
        MK = 3

    psf = dc._normalize_kernel(ndimage.zoom(psf, (MK / MK_source, MK / MK_source, 1))).astype(np.float32)

    if mask:
        mask = [int(x * ratio) for x in mask]

    return image, psf, mask


@jit(float32[:](float32[:], float32[:], float32[:], float32, float32), cache=True, nogil=True)
def _update_image_PAM(u, image, psf, lambd, epsilon=5e-3):
    gradu, TV = divTV(u)
    gradu /= TV
    gradu *= lambd
    gradu += _convolve_image(u, image, psf)
    weight = epsilon * np.amax(u) / np.amax(np.abs(gradu))
    u -= weight * gradu
    np.clip(u, 0, 1, out=u)
    return u


@jit(float32[:](float32[:], float32[:], float32[:], float32, int16, float32), cache=True, nogil=True)
def _loop_update_image_PAM(u, image, psf, lambd, iterations, epsilon, ):
    for itt in range(iterations):
        u = _update_image_PAM(u, image, psf, lambd, epsilon)
        lambd *= 0.99
    return u, psf


@jit(float32[:](float32[:], float32[:], float32[:], float32), cache=True, nogil=True)
def _update_kernel_PAM(u, image, psf, epsilon):
    grad_psf = _convolve_kernel(u, image, psf)
    weight = epsilon * np.amax(psf) / np.maximum(1e-31, np.amax(np.abs(grad_psf)))
    psf -= weight * grad_psf
    psf = dc._normalize_kernel(psf)
    return psf


@jit(float32[:](float32[:], float32[:], float32[:], float32, int16, float32[:], float32[:], float32, boolean),
     cache=True, nogil=True)
def _update_both_PAM(u, image, psf, lambd, iterations, mask_u, mask_i, epsilon, blind):
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
def richardson_lucy_PAM(image, u, psf, lambd, iterations, epsilon=1e-3, mask=None, blind=True):
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

    return u, psf


@utils.timeit
@jit(cache=True)
def deblur_module(pic, filename, dest_path, blur_type, blur_width, noise_reduction_factor, deblur_strength,
                  blur_strength=1,
                  auto_quality=1, ringing_factor=1e-3, refine=False, refine_quality=0, mask=None, debug=False,
                  effect_strength=1, preview=1, refocus=False, psf=None, denoise=False, method="fast", bits=8):
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
    :param noise_reduction_factor: the noise reduction factor lambda. For the `best` method, default is 1000, use 12000
        to 30000 to speed up the convergence. Lower values don't help to reduce the noise, decrease the `ringing_facter` instead.
        for the `fast` method, default is 0.0006, increase up to 0.05 to reduce the noise. This unconsistency must be corrected soon.
    :param ringing_factor: the iterations factor. Typically 1e-3, reduce it to 5e-4 or ever 1e-4 if you see ringing or periodic edges appear.
    :param refine: True or False, decide if the blur kernel should be refined through myopic deconvolution
    :param refine_quality: the number of iterations to perform during the refinement step
    :param mask: the coordinates of the rectangular mask to apply on the image to refine the blur kernel from the top-left corner of the image
        in list [y_top, y_bottom, x_left, x_right]
    :param preview: If you want to fast preview your setting on a smaller picture size, set the downsampling ratio in `previem`. Default : 1
    :param psf: if you already know the PSF kernel, enter it here as an 3D array where the last dimension is the RGB component
    :param denoise: True or False. Perform an initial denoising by Total Variation - Chambolle algorithm before deconvoluting
    :param method: `fast` or `best`. Set the method to deconvolute.
    :return:
    """
    # TODO : refocus http://web.media.mit.edu/~bandy/refocus/PG07refocus.pdf
    # TODO : extract foreground only https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut

    pic = np.asanyarray(pic, dtype=np.float32, order="C")

    # Verify the input and scream like a virgin
    assert (blur_width >= 3), "The PSF kernel is too small !"

    # Set the bit-depth
    samples = 2**bits - 1

    # Assuming 8 bits input, we rescale the RGB values betweem 0 and 1
    pic = np.ascontiguousarray(pic, dtype=np.float32) / samples

    # Choose the RL method
    methods_collection = {
        # "fast": richardson_lucy_PAM, deprecated until update - input are not consistent with those of richardson_lucy_MM
        "best": dc.richardson_lucy_MM
    }

    richardson_lucy = methods_collection[method]

    # Construct a PSF
    if psf == None:
        blur_collection = {
            "gaussian": utils.gaussian_kernel(blur_width, blur_strength),
            "kaiser": utils.kaiser_kernel(blur_width, blur_strength),
            "auto": utils.uniform_kernel(blur_width),
            "uniform": utils.uniform_kernel(blur_width),
            "poisson": utils.poisson_kernel(blur_width, blur_strength),
            "lens": utils.lens_blur(blur_width)
        }

        # TODO http://yehar.com/blog/?p=1495

        psf = blur_collection[blur_type]

    psf = np.dstack((psf, psf, psf)).astype(np.float32)

    if preview != 1:
        print("\nWorking on a scaled picture")
        pic, psf, mask = make_preview(pic, psf, preview, mask)

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

    if denoise:
        # TODO : http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6572468
        u = np.ascontiguousarray(denoise_tv_chambolle(pic, weight=1 / noise_reduction_factor, multichannel=True)).astype(np.float32)
    else:
        u = np.ascontiguousarray(pic.copy()).astype(np.float32)

    if blur_type == "auto":
        print("\n===== BLIND ESTIMATION OF BLUR =====")
        u, psf = process_pyramid(pic, u, psf, noise_reduction_factor, richardson_lucy, ringing_factor,
                                 quality=auto_quality)

        utils.save(u * (2**16 - 1), filename + "-auto-back", dest_path)

    if refine:
        if mask:
            print("\n===== BLIND MASKED REFINEMENT =====")
            u, psf = richardson_lucy(pic, u, psf, noise_reduction_factor, int(10 * refine_quality), ringing_factor,
                                     mask=mask)

        else:
            print("\n===== BLIND UNMASKED REFINEMENT =====")
            u, psf = richardson_lucy(pic, u, psf, noise_reduction_factor, int(10 * refine_quality), ringing_factor)

        utils.save(u * (2**16 - 1), filename + "-refine-back", dest_path)

    if deblur_strength > 0:
        print("\n===== REGULAR DECONVOLUTION =====")
        u, psf = richardson_lucy(pic, u, psf, noise_reduction_factor, int(deblur_strength * 10), ringing_factor,
                                 blind=False, p=0.8)


    # Convert to 16 bits RGB
    u = u * (2**16 - 1)

    # if the picture has been padded to make it odd, unpad it to get the original size
    if odd_hor:
        u = u[:, 1:, ...]

    if odd_vert:
        u = u[1:, :, ...]

    utils.save(u, filename, dest_path)

    return u, psf


if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    # Uncomment the following line if you run into a memory error
    # CPU = 1

    picture = "blured.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # deblur_module(pic, "fast-v4", "kaiser", 0, 0.05, 50, blur_width=11, blur_strength=8)

        # deblur_module(pic, "myope-v4", "kaiser", 10, 0.05, 50, blur_width=11, blur_strength=8, mask=[150, 150 + 256, 600, 600 + 256], refine=True,)


        deblur_module(pic, picture + "-blind-v14-best", dest_path, "auto", 5, 12000, 2,
                      mask=[150, 150 + 256, 600, 600 + 256],
                      refine=True,
                      refine_quality=2,
                      auto_quality=1,
                      ringing_factor=1e-3,
                      method="best",
                      debug=True)



        """
        deblur_module(pic, picture + "-blind-v10-fast", dest_path, "auto", 5, 0.05, 0,
                      mask=[150, 150 + 512, 600, 600 + 512],
                      refine=True,
                      refine_quality=50,
                      auto_quality=2,
                      method="fast",
                      debug=True)
                      
        """
        pass

    picture = "Shoot-Sienna-Hayes-0042-_DSC0284-sans-PHOTOSHOP-WEB.jpg"
    with Image.open(join(source_path, picture)) as pic:
        """
        deblur_module(pic, picture + "test-v7-norme-2", dest_path, "kaiser", 9, 5000, 0,
                      mask=[318, 357 + 256, 357, 357 + 256],
                      denoise=False,
                      blur_strength=8,
                      refine=True,
                      refine_quality=2,
                      auto_quality=2,
                      preview=1,
                      debug=True,
                      method="best",
                      )


        """
        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:

        mask = [631+256, 631 + 1024+256, 2826, 2826 + 1024]
        """
        deblur_module(pic, picture + "test-v7", dest_path, "auto", 9, 5000, 0,
                      mask=mask,
                      refine=True,
                      refine_quality=1,
                      auto_quality=1,
                      preview=1,
                      debug=True,
                      method="best",
                      )
        """
        pass

    picture = "IMG_9584-900.jpg"
    with Image.open(join(source_path, picture)) as pic:
        """
        mask = [201, 201+128, 167, 167+128]
        deblur_module(pic, picture + "test-v7-2-3", dest_path, "auto", 3, 30000, 2,
                      denoise=False,
                      mask=mask,
                      refine=True,
                      refine_quality=2,
                      auto_quality=2,
                      preview=1,
                      method="best"
        )
        """

        pass

    # JPEG input example

    picture = "153412.jpg"
    with Image.open(join(source_path, picture)) as pic:

        mask = [1484, 1484 + 256, 3228, 3228 + 256]
        deblur_module(pic, picture + "-blind-v14-best", dest_path, "auto", 7, 30000, 0,
                      mask=mask,
                      refine=True,
                      refine_quality=2,
                      preview=1,
                      auto_quality=1,
                      method="best",
                      debug=True)




        pass

    # TIFF input example

    """
    source_path = "/home/aurelien/Exports/2017-11-19-Shoot Fanny Wong/export"
    picture = "Shoot Fanny Wong-0146-_DSC0426--PHOTOSHOP.tif"
    pic = tifffile.imread(join(source_path, picture))

    mask = [1914, 1914 + 722, 718, 1484 + 1012]
    deblur_module(pic, picture + "-blind-v11-best-3", dest_path, "auto", 11, 30000, 0.5,
                  mask=mask,
                  refine=True,
                  refine_quality=0.5,
                  preview=0.25,
                  auto_quality=0.5,
                  method="best",
                  bits=16)

    """