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
from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import color
import numpy as np

from lib import utils

import numba

@utils.timeit
@numba.jit
def processing(pic):
    
    pic = np.array(pic).astype(float)
                
    #Make a L copy of the original data 
    L_backup = color.rgb2lab(pic / 255)[..., 0]
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(10, 8)
                
    # Make a Richardson- Lucy deconvolution on the RGB signal
    pic = utils.richardson_lucy(pic, psf, 3)
    
    # Convert to LAB
    pic = color.rgb2lab(pic / 255)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(10, 12)
    
    # Make an additional Richardson- Lucy deconvolution on L channel
    pic[..., 0] = utils.richardson_lucy(pic[..., 0], psf, 1.5)
    
    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)

    return pic

if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    images = [f for f in listdir(source_path) if isfile(join(source_path, f))]

    # The strength of the unsharp mask
    amount = 0.5
    
    for picture in images:
        
     with Image.open(join(source_path, picture)) as pic:
        
        pic = processing(pic)
        
        with Image.fromarray(pic) as output:
    
            output.save(join(dest_path, picture),
                        format="jpeg",
                        optimize=True,
                        progressive=True,
                        quality=90)
   