'''
Created on 30 avr. 2017

@author: aurelien

This script shows an implementation of the bilateral filter in LAB space.

Applying it on A and B channels is useful to remove the fringing due to the 
chromatic aberrations.

'''
from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import color
import numba

from lib import utils
import numpy as np


@utils.timeit
@numba.jit
def process(pic):
    # Open RGB image in float mode
    pic = np.array(pic).astype(float)

    # Convert to LAB
    pic = color.rgb2lab(pic / 255)

    # Compute a bilateral filter on A channel
    A = utils.bilateral_filter(pic[..., 2], 8, 18, 12)

    # USM formula
    pic[..., 2] = pic[..., 2] - (pic[..., 2] - A) * amount

    # Convert back to 8 bits RGB before saving
    pic = (color.lab2rgb(pic) * 255).astype(np.uint8)

    return pic


if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/bilateral-LAB"
    images = [f for f in listdir(source_path) if isfile(join(source_path, f))]

    amount = 1.5

    for picture in images:
        with Image.open(join(source_path, picture)) as pic:

            pic = process(pic)

            with Image.fromarray(pic) as output:

                output.save(join(dest_path, picture),
                            format="jpeg",
                            optimize=True,
                            progressive=True,
                            quality=90)
