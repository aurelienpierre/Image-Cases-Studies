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
from PIL import Image, ImageDraw
from scipy import misc
from skimage import color
from scipy.signal import fftconvolve
from multiprocessing import Pool


import numba

from lib import utils
import numpy as np
from os.path import isfile, join

@numba.jit(cache=True)
def nabla(I):
    # http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    G = np.zeros((I.shape[0], I.shape[1], 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G

@numba.jit(cache=True)
def nablaT(G):
    I = np.zeros((G.shape[0], G.shape[1]), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1: ] += G[:, :-1, 0]
    I[:-1]    -= G[:-1, :, 1]
    I[1: ]    += G[:-1, :, 1]
    return I

@numba.jit(cache=True)
def anorm(x):
    '''Calculate L2 norm over the last array dimension'''
    return np.sqrt(np.square(x).sum(-1))

@numba.jit(cache=True)
def project_nd(P, r):
    '''perform a pixel-wise projection onto R-radius balls'''
    nP = np.maximum(1.0, anorm(P)/r)
    return P / nP[...,np.newaxis]

@numba.jit(cache=True)
def regularization(image, im_deconv, lambd):
    """Total variation regularisation term as described in :
    Richardson-Lucy Deblurring for Scenes Under A Projective Motion Path
    Yu-Wing Tai  Ping Tan  Michael S. Brown
    http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf"""

    # Compute total variation
    Grad_R_TV = - nablaT(nabla(nablaT(project_nd(nabla(im_deconv), 1.0))))

    # regularize
    im_deconv /= (1 - lambd * Grad_R_TV)

    # Clip out of range values
    factor = im_deconv.max()
    im_deconv = im_deconv + np.clip(image - im_deconv, -lambd*factor, lambd*factor)

    return im_deconv

@numba.jit(cache=True)
def deconvolution(image, im_deconv, psf, psf_mirror):
    relative_blur = (image / fftconvolve(im_deconv, psf, 'same'))
    im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')
    return im_deconv

def richardson_lucy(image, psf, lambd, iterations):
    """Richardson-Lucy deconvolution.

    Adapted from skimage.restore module

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    lambd : float
       Lambda parameter of the total Variation regularization

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    .. [3] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [4] http://www.groupes.polymtl.ca/amosleh/papers/ECCV14.pdf
    """

    # Pad the image with symmetric data to avoid border effects
    image = np.pad(image, (iterations, iterations), mode="symmetric")

    im_deconv = 0.5 * np.ones_like(image)
    psf_mirror = psf[::-1, ::-1]

    for i in range(iterations):
        # Richardson-Lucy actual deconvolution
        im_deconv = deconvolution(image, im_deconv, psf, psf_mirror)

        if lambd !=0:
            # Noise regularization
            im_deconv = regularization(image, im_deconv, lambd)
            lambd /= 2


    # Unpad image
    im_deconv = im_deconv[iterations :-iterations , iterations :-iterations ]
    image = image[iterations:-iterations, iterations:-iterations]

    # Compute error
    delta = np.absolute(image - im_deconv)
    error = np.linalg.norm(delta, 2)
    print("Difference :", error)

    return im_deconv


@utils.timeit
def processing_FAST(pic):
    
    pic = np.array(pic).astype(float)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(10, 8)
                
    # Make a Richardson- Lucy deconvolution on the RGB signal
    with Pool(processes=3) as pool:
        pic = np.dstack(pool.starmap(
            richardson_lucy,
            [(pic[..., i], psf, 10, 50) for i in range(3)]
            )
        )

    # Convert to LAB
    pic = color.rgb2lab(pic / 255)

    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)

    return pic

@utils.timeit
def processing_BEST(pic):
    
    sampling = 2.0
    
    # Open the picture
    pic = np.array(pic).astype(np.uint8)
    
    #Oversample the image with Lanczos interpolation
    pic= misc.imresize(pic, sampling, interp="lanczos", mode="RGB").astype(float)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(20, 8)


    # Make a Richardson- Lucy deconvolution on the RGB signal
    with Pool(processes=3) as pool:
        pic = np.dstack(pool.starmap(
            richardson_lucy,
            [(pic[..., i], psf, 10, 50) for i in range(3)]
        )
        )
    
    # Convert to LAB
    pic = color.rgb2lab(pic / 255)

    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)
    
    #Resize back to original picture
    pic = misc.imresize(pic, 1/sampling, interp="lanczos")
    
    return pic


def save(pic, name):
    with Image.fromarray(pic) as output:

        # Draw the mask
        #draw = ImageDraw.Draw(output)
        #draw.rectangle([(680, 252), (680 + 320, 252 + 734)], fill=None, outline=128)
        #del draw

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

            pic_best = processing_BEST(pic)
            save(pic_best, "best")


            pic_fast = processing_FAST(pic)
            save(pic_fast, "fast")

            # Richardson extrapolation assuming order 1 convergence :
            # https://en.wikipedia.org/wiki/Richardson_extrapolation
            pic_extrapol = - color.rgb2lab(pic_fast / 255) + 2 * color.rgb2lab(pic_best / 255)
            pic_extrapol = (color.lab2rgb(pic_extrapol) * 255).astype(np.uint8)

            save(pic_extrapol, "extrapol")
