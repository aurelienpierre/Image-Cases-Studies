# -*- coding: utf-8 -*-k
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
from skimage.transform import resize


from lib import tifffile
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

    images = [1.]
    kernels = [psf_size]


    while kernels[-1] > 3:
        kernels.append(int(np.ceil(kernels[-1] / np.sqrt(2))))
        images.append(images[-1] / np.sqrt(2))

        if kernels[-1] % 2 == 0:
            kernels[-1] -= 1

        if kernels[-1] < 3:
            kernels[-1] = 3

    return images, kernels


from skimage.restoration import denoise_tv_chambolle

@utils.timeit
def deblur_module(pic, filename, dest_path, blur_width, confidence=10, tolerance=1, quality="normal", bits=8,
                  mask=None, display=True, blur="static", preview=False, p=1, order=2, norm=1, priority=0, mask_size=255, iterations=200, refocus=False):
    """
    API to call the debluring process

    :param pic: an image memory object, from PIL or tifffile
    :param filename: string, the name of the file to save
    :param dest_path: string, the path where to save the file
    :param blur_width: integer, the diameter of the blur e.g. the size of the PSF
    :param confidence: float, default 1, max 100, set the confidence you have in your sample. For example, on noisy pictures,
    use 1 to 10. For a clean low-ISO picture, you can go all the way to 100. A low factor will reduce the convergence, a high
    factor will allow more noise amplification.
    :param tolerance: float, between 0 and 100. The amount of error you can accept in the solution in %.
    :param bits: integer, default is 8 meaning the input image is encoded with 8 bits/channel. Use 16 if you input 16 bits
    tiff files.
    :param mask: list of 2 integers, the center of th region on which the blur will be estimated to speed-up the process.
    :param display: Pop-up a control window at the end of the blur estimation to check the solution before runing it on
    the whole picture
    :param p: float, the power of the Total Variation used to regularize the deblurring. Set > 2 to increase the convergence rate but this might favor the blurry picture as well.
    It will be refined during the process anyway.
    :return:
    """
    # TODO : refocus http://web.media.mit.edu/~bandy/refocus/PG07refocus.pdf
    # TODO : extract foreground only https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut

    pic = np.ascontiguousarray(pic, dtype=np.float32)

    # Extrapad for safety
    pic = pad_image(pic, (1, 1)).astype(np.float32)

    # Set the bit-depth
    samples = 2**bits - 1

    # Rescale the RGB values between 0 and 1
    pic = pic / samples

    # Undo the gamma corrector
    pic = pic ** (1/2.2)

    # Map the quality to gradient descent step
    if quality == "normal":
        step = 1e-3
    elif quality == "high":
        step = 5e-4
    elif quality == "veryhigh":
        step = 1e-4
    elif quality == "low":
        step = 5e-3


    # Blur verifications
    if blur_width < 3:
        raise ValueError("The blur width should be at least 3 pixels.")
    elif blur_width % 2 == 0:
        raise ValueError("The blur width should be odd. You can use %i." % (blur_width + 1))

    #TODO : automatically evaluate blur size : https://www.researchgate.net/publication/257069815_Blind_Deconvolution_of_Blurred_Images_with_Fuzzy_Size_Detection_of_Point_Spread_Function


    # Get the dimensions once for all
    MK = blur_width # PSF size
    M = pic.shape[0] # Image height
    N = pic.shape[1] # Image width
    C = 3 # RGB channels


    # Define a minimum mask size for the blind deconvolution
    if mask is None:
        # By default, set the mask in the center of the picture
        mask = [M//2, N//2]

    # Create the coordinates of the masking box
    top = mask[0] - mask_size//2
    bottom = mask[0] + mask_size//2
    left = mask[1] - mask_size//2
    right = mask[1] + mask_size//2

    print("Mask size :", (bottom - top + 1), "×", (right - left + 1))

    if top > 0 and bottom < M and left > 0 and right < N:
       pass
    else:
       raise ValueError("The mask is outside the picture boundaries. Move its center inside or reduce the blur size.")


    # Adjust the blur type.
    # For motion blur, we enforce the RGB of the PSF to have the same coefficients
    # This is to help the solver converge.
    if blur == "static":
        correlation = False
    elif blur == "motion":
        correlation = True


    # Rescale the tolerance
    tolerance /= 100.

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

    # Construct a uniform PSF : ones everywhere
    psf = utils.uniform_kernel(blur_width)
    psf = np.dstack((psf, psf, psf))

    # Build the pyramid
    images, kernels = build_pyramid(blur_width, confidence)

    # Convergence flag
    # When a pyramid step does not converge inside the amount of iterations fixed
    # it's usually because the blur size is ill-chosen
    # so the convergence flag is raised and the deconvolution is stopped
    convergence_flag = False

    try:

        # Launch the pyramid deconvolution
        for case in ["blind", "non-blind"]:
            print("\n===== %s DECONVOLUTION =====" % case)

            deblured_image = pic.copy()

            # The algorithm is designed to make lambda converge no matter what
            # But a kick on the right path is nice
            lambd = confidence * 1000
            p_temp = p
            norm_temp = norm

            for i, k in zip(reversed(images), reversed(kernels)):
                # Still not sure if we should process the pyramid in a non-blind setup
                if True:#case == "blind" or i == 1.:
                    print("======== Pyramid step %1.3f ========" % i)

                    # Compute the new sizes of the mask
                    temp_top = int(i * top)
                    temp_bottom = int(i * bottom)
                    temp_left = int(i * left)
                    temp_right = int(i * right)

                    # Make sure the mask dimensions will be odd and square
                    if int(temp_bottom - temp_top) % 2 == 0:
                        if int(temp_bottom - temp_top) < int(temp_right - temp_left):
                            temp_bottom += 1
                        elif int(temp_bottom - temp_top) > int(temp_right - temp_left):
                            temp_top += 1
                        else:
                            temp_top -= 1

                    if int(temp_right - temp_left) % 2 == 0:
                        if int(temp_bottom - temp_top) < int(temp_right - temp_left):
                            temp_left += 1
                        elif int(temp_bottom - temp_top) > int(temp_bottom - temp_top):
                            temp_right += 1
                        else:
                            temp_right -= -1

                    # Compute the new size of the picture
                    temp_width = int(np.floor(i * N))
                    temp_height = int(np.floor(i * M))

                    # Ensure oddity on the picture
                    if temp_width % 2 == 0:
                        temp_width += 1
                    if temp_height % 2 == 0:
                        temp_height += 1

                    shape = (temp_height, temp_width, 3)

                    # Resize blured, deblured images and PSF from previous step
                    temp_blurry_image = resize(pic, shape, order=3, mode="edge", preserve_range=True).astype(np.float32)
                    deblured_image = resize(deblured_image, shape, order=3, mode="edge", preserve_range=True).astype(np.float32)

                    if case == "blind":
                        psf_copy = resize(psf, (k, k, 3), order=3, mode="edge", preserve_range=True).astype(np.float32)
                        dc.normalize_kernel(psf_copy, k)
                    else:
                        psf_copy = psf.copy()
                        k = kernels[0]

                    # Extra safety padding - Remember the gradient is not evaluated on borders
                    temp_blurry_image = pad_image(temp_blurry_image, (1, 1)).astype(np.float32)
                    deblured_image = pad_image(deblured_image, (1, 1)).astype(np.float32)

                    # Pad for FFT
                    pad = int(np.floor(k / 2))

                    # Debug
                    print("Image size", temp_blurry_image.shape)
                    print("u size", deblured_image.shape)
                    print("Mask size", (temp_bottom - temp_top), (temp_right - temp_left))
                    print("PSF size", psf_copy.shape)

                    # Disallow tolerance on lower pyramid sizes
                    # This because creating noise while downscaled will result in smudges when upscaled
                    if i == 1.:
                        tolerance_temp = tolerance
                    else:
                        tolerance_temp = 0

                    # Make a blind Richardson-Lucy deconvolution on the RGB signal
                    if case == "blind":
                        deblured_image[temp_top - 1:temp_bottom + 1, temp_left - 1:temp_right + 1, ...] = dc.richardson_lucy_MM(
                                                temp_blurry_image[temp_top - 1:temp_bottom + 1, temp_left - 1:temp_right +1, ...],
                                                deblured_image[temp_top - pad - 1:temp_bottom + pad + 1, temp_left - pad - 1:temp_right + pad + 1, ...],
                                                psf_copy,
                                                pad+1, temp_bottom - temp_top - pad - 1, pad+1, temp_bottom - temp_top - pad-1,
                                                0,
                                                temp_bottom - temp_top + 2,
                                                temp_right - temp_left + 2,
                                                3,
                                                k, iterations, step, lambd, blind=True, p=p_temp, correlation=correlation, order=order, norm=2, priority=0, refocus=refocus)
                        # Update the PSF
                        psf = psf_copy.copy()

                    elif case != "blind" and preview:
                        deblured_image[temp_top - 1:temp_bottom + 1, temp_left - 1:temp_right  + 1, ...] = dc.richardson_lucy_MM(
                                                temp_blurry_image[temp_top - 1:temp_bottom + 1, temp_left - 1:temp_right + 1, ...],
                                                deblured_image[temp_top - pad - 1:temp_bottom + pad + 1, temp_left - pad - 1:temp_right + pad + 1, ...],
                                                psf_copy,
                                                pad+1, temp_bottom - temp_top - pad - 1, pad+1, temp_bottom - temp_top - pad-1,
                                                tolerance_temp,
                                                temp_bottom - temp_top + 2,
                                                temp_right - temp_left + 2,
                                                3,
                                                k, iterations, step, lambd, blind=False, p=p_temp,order=order, norm=2, priority=priority, refocus=refocus)
                    else:
                    # Pad for FFT
                        deblured_image = pad_image(deblured_image, (pad, pad)).astype(np.float32)
                        deblured_image[pad:-pad, pad:-pad, ...] = dc.richardson_lucy_MM(
                                                temp_blurry_image,
                                                deblured_image,
                                                psf_copy,
                                                pad+1, temp_bottom - temp_top - pad - 1, pad+1, temp_bottom - temp_top - pad-1,
                                                tolerance_temp,
                                                temp_height + 2,
                                                temp_width + 2,
                                                3,
                                                k, iterations, step, lambd, blind=False, p=p_temp, order=order, norm=2, priority=priority, refocus=refocus)

                        # Unpad FFT because this image is resized/reused the next step
                        deblured_image  = deblured_image[pad:-pad, pad:-pad, ...]

                    if convergence_flag:
                        raise RuntimeError("The optimization didn't converge. It usually means your blur size is ill-chosen.")

                    # Remove the extra safety padding
                    temp_blurry_image = temp_blurry_image[1:-1, 1:-1, ...]
                    deblured_image = deblured_image[1:-1, 1:-1, ...]

                    # Update the norm
                    norm_temp /= 2

                    k_prec = k

            # Display the control preview
            if display and case=="blind":
                psf_check = (psf - np.amin(psf)) / (np.amax(psf) - np.amin(psf))
                plt.imshow(psf_check, interpolation="lanczos", filternorm=1, aspect="equal", vmin=0, vmax=1)
                plt.show()
                plt.imshow((deblured_image[top:bottom, left:right, ...]*255).astype(np.uint8), interpolation="lanczos", filternorm=1, aspect="equal", vmin=0, vmax=255)
                plt.show()

    except KeyboardInterrupt:
        # Nasty trick to be able to hard-shutdown the iterations and still get the output
        # Don't try this at home
        # Seriously, don't.
        pass


    # Clip extreme values
    np.clip(deblured_image, 0., 1., out=deblured_image)

    # Redo the gamma corrector
    deblured_image = deblured_image ** (2.2)

    # Convert to 16 bits RGB
    deblured_image = deblured_image * (2 ** 16 - 1)

    # Save the pic
    if preview:
        filename = filename + "-preview"
        deblured_image = deblured_image[top:bottom, left:right, ...]
    else:
        # if the picture has been padded to make it odd, unpad it to get the original size
        if odd_hor:
            deblured_image = deblured_image[:, 1:, ...]
        if odd_vert:
            deblured_image = deblured_image[1:, :, ...]

        # Remove the extra pad
        deblured_image = deblured_image[1:-1, 1:-1, ...]

    utils.save(deblured_image, filename, dest_path)

if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    picture = "6Q1B9891.jpg"
    with Image.open(join(source_path, picture)) as pic:
        """
        mask = [833, 942]
        deblur_module(pic, picture + "-v28", dest_path, 13, mask=mask, display=False, tolerance=0.1, confidence = 10, quality ="normal", preview=True, p=1, blur="motion", order=2, norm=1, priority=1, iterations=1000, mask_size=511)
        """
        pass

    picture = "blured.jpg"
    with Image.open(join(source_path, picture)) as pic:

        mask = [584, 795]
        deblur_module(pic, picture + "-v29", dest_path, 7, mask=mask, display=False, tolerance=0.1, quality ="normal", preview=False, p=1, blur="static", order=2, norm=1, priority=1, iterations=200)

        pass

    picture = "IMG_9584-900.jpg"
    with Image.open(join(source_path, picture)) as pic:
        """
        deblur_module(pic, picture + "-v28-L2-1", dest_path, 3, display=True, tolerance=1, preview=False, order=1, norm=2, p=2, priority=0.)
        deblur_module(pic, picture + "-v28-L2-2", dest_path, 3, display=True, tolerance=1, preview=False, order=2, norm=2, p=2, priority=0.)
        deblur_module(pic, picture + "-v28-L1-1", dest_path, 3, display=True, tolerance=1, preview=False, order=1, norm=1, p=2, priority=0.)
        deblur_module(pic, picture + "-v28-L1-2", dest_path, 3, display=True, tolerance=1, preview=False, order=2, norm=1, p=2, priority=0.)
        """
        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [500, 4050]
        #deblur_module(pic, picture + "-v28", dest_path, 3, mask=mask, tolerance=10, display=False, preview=True, priority=-1.0, mask_size=511, norm=1.)
        pass

    picture = "P1030302.jpg"
    with Image.open(join(source_path, picture)) as pic:
        #mask = [1645, 482]
        #deblur_module(pic, picture + "-v29", dest_path, 45, display=False, tolerance=0., preview=False, quality="normal", priority=0, mask_size=511)
        pass

    picture = "153412-inpainting.jpg"
    with Image.open(join(source_path, picture)) as pic:
        mask = [1800, 3650]
        #deblur_module(pic, picture + "-v28", dest_path, 3, mask=mask, display=True, mask_size=811, preview=True, tolerance=100., priority=1., iterations=100, norm=1, order=2, refocus=False)
        pass

    # TIFF input example
    source_path = "/home/aurelien/Exports/2017-11-19-Shoot Fanny Wong/export"
    picture = "Shoot Fanny Wong-0146-_DSC0426--PHOTOSHOP.tif"
    #pic = tifffile.imread(join(source_path, picture))
    mask = [1914, 1484]
    #deblur_module(pic, picture + "-blind-v18", dest_path, 5, mask=mask, bits=16)


