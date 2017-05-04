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
from os import listdir
from os.path import isfile, join
from skimage import color
from scipy import misc
import numpy as np

from lib import utils

import numba

@utils.timeit
@numba.jit(cache=True)
def processing_FAST(pic):
    
    pic = np.array(pic).astype(float)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(10, 8)
    
    # Generate a mask around the face, excluding the background
    mask = np.zeros_like(pic[..., 0])
    
    # Draw a 510×734 pixels mask beginning at the coordinate [252, 608] (corner)
    mask[252:252+734, 680:680+320] = 1
                
    # Make a Richardson- Lucy deconvolution on the RGB signal
    pic = utils.richardson_lucy(pic, psf, 3, iterations=100)
    
    # Convert to LAB
    pic = color.rgb2lab(pic / 255)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(10, 12)
    
    # Make an additional Richardson- Lucy deconvolution on L channel
    L = utils.richardson_lucy(pic[..., 0], psf, 1, weight=mask, iterations=100)
    
    # USM formula
    pic[..., 0] = pic[..., 0] + (L - pic[..., 0]) * amount
    
    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)

    return pic

@utils.timeit
@numba.jit(cache=True)
def processing_BEST(pic):
    
    sampling = 2.0
    
    # Open the picture
    pic = np.array(pic).astype(np.uint8)
    
    #Oversample the image with Lanczos interpolation
    pic= misc.imresize(pic, sampling, interp="lanczos", mode="RGB").astype(float)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(20, 8)
    
    # Generate a mask around the face, excluding the background
    mask = np.zeros_like(pic[..., 0])
    
    # Draw a 510×734 pixels mask beginning at the coordinate [252, 608] (corner)
    mask[round(252*sampling) : round((252+734)*sampling) , 
         round(680*sampling) : round((680+320)*sampling) ] = 1
                
    # Make a Richardson- Lucy deconvolution on the RGB signal
    pic = utils.richardson_lucy(pic, psf, 3, iterations=30)
    
    # Convert to LAB
    pic = color.rgb2lab(pic / 255)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(20, 24)
    
    # Make an additional Richardson- Lucy deconvolution on L channel
    L = utils.richardson_lucy(pic[..., 0], psf, 1, weight=mask, iterations=50)
    
    # USM formula
    pic[..., 0] = pic[..., 0] + (L - pic[..., 0]) * amount
    
    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)
    
    #Resize back to original picture
    pic = misc.imresize(pic, 1/sampling, interp="lanczos")
    
    return pic

def save(pic, name):
    with Image.fromarray(pic) as output:
            
            # Draw the mask
            draw = ImageDraw.Draw(output)
            draw.rectangle([(680, 252), (680+320, 252+734)], fill=None, outline=128)
            del draw
    
            output.save(join(dest_path, picture + "-" + name),
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
        
        # The strength of the unsharp mask
        amount = 2
        pic_best = processing_BEST(pic)
        
        save(pic_best, "best")
        
        del pic_best
        
        
        # The strength of the unsharp mask
        amount = 1
        pic_fast = processing_FAST(pic)
        
        save(pic_fast, "fast")
   