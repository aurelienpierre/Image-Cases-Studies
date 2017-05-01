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
import pyximport

from lib import utils
pyximport.install()

if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/bilateral-unsharp-mask"

    images = [f for f in listdir(source_path) if isfile(join(source_path, f))]

    # The strength of the unsharp mask
    amount = 0.5

    for picture in images:
        with Image.open(join(source_path, picture)) as pic:

            # Create a LAB/RGB image object
            pic = utils.image_open(pic)

            # Compute a bilateral filter on L channel
            L = utils.bilateral_filter(pic.L, 10, 6.0, 3.0)

            # USM formula
            pic.L = pic. L + (pic.L - L) * amount

            with Image.fromarray(pic.RGB) as output:

                output.save(join(dest_path, picture),
                            format="jpeg",
                            optimize=True,
                            progressive=True,
                            quality=90)
