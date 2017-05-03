'''
Created on 30 avr. 2017

@author: aurelien

This script shows an implementation of the unsharp mask (sharpening method ) 
with a bilateral filter instead of the classic Gaussian filter

This is useful to increase the apparent sharpness of an image without adding
halos.
'''
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import color
import numpy as np
from lib import utils

import numba

@utils.timeit
@numba.jit
def process(pic):
     # Open RGB image in float mode
    pic = np.array(pic).astype(float)
    
    # Convert to LAB
    pic = color.rgb2lab(pic / 255)
    
    # Compute a bilateral filter on L channel
    L = utils.bilateral_filter(pic[..., 0], 10, 6, 3)

    # USM formula
    pic[..., 0] = pic[..., 0] + (pic[..., 0] - L) * amount
    
    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)
    
    return pic

if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/bilateral-unsharp-mask"

    images = [f for f in listdir(source_path) if isfile(join(source_path, f))]

    # The strength of the unsharp mask
    amount = 0.5

    for picture in images:
        with Image.open(join(source_path, picture)) as pic:

            pic = process(pic)
        
            with Image.fromarray(pic) as output:
        
                output.save(join(dest_path, picture),
                            format="jpeg",
                            optimize=True,
                            progressive=True,
                            quality=90)
