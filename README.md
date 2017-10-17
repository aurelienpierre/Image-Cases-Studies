 # Image Cases Studies
Python prototypes of image processing methods

## Presentation

### Motivation

This collection of scripts is intended to prototype methods and functionalities that
could be useful in [darktable](https://github.com/darktable-org/darktable) and
show proofs of concept.

### How it's made

It's written in Python 3, and relies on PIL (Python Image Library) for the I/O, Numpy for the arrays
operations, and Numba to optimize the execution time. Heavy arrays operations 
are parallelized through multiprocesses but can be run serialized as well.


Every function is timed natively, so you can benchmark performance. 

### What's inside

For now, we have :

* Filters :
    * Gaussian blur
    * Bessel blur (Kaiser denoising)
    * bilateral filter
    * unsharp mask
    * Richardson-Lucy blind and non-blind deconvolution with Total Variation regularization
* Windows/Kernels : (*for convolution and frequential analysis*)
    * Poisson/exponential
    * Kaiser-Bessel
    * Gauss
    
A collection of test pictures is in `img` directory and the converted pictures
are in `img` subfolders. The built-in functions are in the `lib.utils` module.
    
### Current prototypes

#### Richardson-Lucy deconvolution

In theory, blurred and noisy pictures can be perfectly sharpened if we perfectly 
know the [*Point spread function*](https://en.wikipedia.org/wiki/Point_spread_function) 
of their maker. In practice, we can only estimate it.
One of the means to do so is the [Richardson-Lucy deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution).

The Richardson-Lucy algorithm used here is  modified to implement [Total Variation regularization
](http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf). It can be run in a non-blind fashion (when the PSF is known)
or in a blind one to determine the PSF iteratively from an initial guess.

##### Blurred original :
![alt text](img/blured.jpg)

##### After (fast algorithm - 35 s - 50 iterations - Non blind):
This takes in input an user-defined PSF guessed by trial and error.
![alt text](img/richardson-lucy-deconvolution/blured-fast-v3.jpg)

##### After (myope algorithm - 73 s - 50 iterations - Semi-Blind refinement):
This takes in input an user-defined PSF guessed by trial and error but will refine it every iteration on a 256×256 px sampling patch.
(drawn in red here).
![alt text](img/richardson-lucy-deconvolution/blured-myope-v5.jpg)

##### After (blind algorithm - 132 s - 100 iterations - Blind):
This takes no input and will build the SPF along from scratch. 
A balance between the masked zone weight and the whole image weight in the computation can be adjusted.
![alt text](img/richardson-lucy-deconvolution/blured-blind-v7.jpg)


## Installation

It's not recommended to install this *unstable* framework on your Python environnement, but rather to build
its modules and use it from its directory.

    python setup.py build_ext --inplace

On Linux systems, if you have Python 2 and 3 interpreters installed together, you may run :

    python3 setup.py build_ext --inplace
    
The Python interpreter should be in the 3.x serie (3.5 is best).
    
Unfortunately, the setup file has been reported defective so in most cases, the dependencies will
not be automatically installed.

To solve this problem until an elegant solution is found, the simpliest way is to first install the [Anaconda Python distribution](https://www.anaconda.com/download/)
which is a bundle of Python packages for scientific computation and signal processing.

Then, ensure the following packages are installed :

    PIL (known as pillow)
    numba
    scipy 
    numpy (normally included into scipy)
    sympy
    skimage
    pyfftw

    
## Use

### In console

Execute :

```shell
 python3 richardson_lucy_deconvolution.py 
```

Import the required Python packages : 

```python
from lib import utils # if you are working directly in the package directory
from PIL import Image 
import numpy as np
from skimage import color
```
    
Load an image as a 3D RGB array:

```python
with Image.open("path/image") as pic:

        pic = np.array(pic).astype(float)
```
    
Set/Reset RGB channels 

```python
pic[..., 0] = numpy.array([...]) # sets the R channel with a 2D numpy array
pic[..., 1] = numpy.array([...]) # sets the G channel with a 2D numpy array
pic[..., 2] = numpy.array([...]) # sets the B channel with a 2D numpy array
```
    
    
Blur a channel : 

    i = # 0, 1 or 2
    pic[..., i] = utils.bilateral_filter(pic[..., i], 10, 6.0, 3.0)
    
Save the picture :
    
```python
with Image.fromarray(pic) as output:

    output.save("file.jpg")
```
    
See the scripts in the root directory for real examples.
