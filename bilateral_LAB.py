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
import pyximport

from lib import utils
pyximport.install()

if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/bilateral-LAB"
    images = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    
    amount = 1.5

    for picture in images:
        with Image.open(join(source_path, picture)) as pic:

            # Create a LAB/RGB image object
            pic = utils.image_open(pic)

            # Compute a bilateral filter on A channel
            A = utils.bilateral_filter(pic.A, 8, 18.0, 18.0)
            
            # USM formula
            pic.A = pic.A - (pic.A - A) * amount

            with Image.fromarray(pic.RGB) as output:

                output.save(join(dest_path, picture),
                            format="jpeg",
                            optimize=True,
                            progressive=True,
                            quality=90)
