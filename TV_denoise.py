'''
Created on 29 oct. 2017

@author: aurelien

'''

from os.path import join

import numpy as np
import pyfftw
import scipy
from PIL import Image
from scipy.signal import convolve
from skimage.restoration import denoise_tv_chambolle

pyfftw.interfaces.cache.enable()
scipy.signal.signaltools.fftn = pyfftw.interfaces.scipy_fftpack.fftn
scipy.signal.signaltools.ifftn = pyfftw.interfaces.scipy_fftpack.ifftn
scipy.fftpack = pyfftw.interfaces.scipy_fftpack


def auto_denoise(image, k, b):
    """
    Implement an automatic estimation of the best TV-denoising lambda parameter based on the technique of
    Baoxian Wang, Baojun Zhao, Chenwei Deng and Linbo Tang, 2012
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6572468

    :param image:
    :return:
    """
    # Normalized for 8 bits JPGS
    image = np.ascontiguousarray(image, np.float32) / 255
    TV = divTV(image)
    MTV = np.linalg.norm(TV / TV.size)

    sigma = estimate_sigma(image, multichannel=True)
    print("Noise level :", sigma)

    sigma = k + MTV + b
    print("Noise level :", sigma)
    lambd_A = 0.5 * sigma
    lambd_B = 2 * sigma

    tau = 0.18
    epsilonn = 0.001

    lambd_L = lambd_B - tau * (lambd_B - lambd_A)
    lambd_H = lambd_A + tau * (lambd_B - lambd_A)

    while (lambd_B - lambd_A) > epsilonn:
        f_H = MinDenoise(image, lambd_H, sigma)
        f_L = MinDenoise(image, lambd_L, sigma)

        if f_L < f_H:
            lambd_B = lambd_H
            f_H = f_L
            lambd_L = lambd_B + tau * (lambd_B - lambd_A)
            MinTV(image, lambd_L)

    print(MTV)

    return image, lambd


def MinTV(image, lambd):
    a = np.linalg.norm(image - denoise_tv_chambolle(image, weight=lambd, multichannel=True))
    b = 2 * lambd * np.amax(np.abs(image))
    return np.minimum(a, b)


def MinDenoise(image, lambd, sigma):
    return np.linalg.norm(image - \
                          denoise_tv_chambolle(image, weight=lambd, multichannel=True)) \
           - image.size * sigma


"""

http://eeweb.poly.edu/iselesni/lecture_notes/TVDmm/TVDmm.pdf


function [x, cost] = tvd_mm(y, lam, Nit)
% [x, cost] = tvd_mm(y, lam, Nit)
% Total variation denoising using majorization-minimization
% and banded linear systems.
%
% INPUT
%   y - noisy signal
%   lam - regularization parameter
%   Nit - number of iterations
%
% OUTPUT
%   x - denoised signal
%   cost - cost function history
%
% Reference
% ’On total-variation denoising: A new majorization-minimization
% algorithm and an experimental comparison with wavalet denoising.’
% M. Figueiredo, J. Bioucas-Dias, J. P. Oliveira, and R. D. Nowak.
% Proc. IEEE Int. Conf. Image Processing, 2006.
% Ivan Selesnick, selesi@nyu.edu, 2011
% Revised 2017
y = y(:);                                              % Make column vector
cost = zeros(1, Nit);                                  % Cost function history
N = length(y);
I = speye(N);
D = I(2:N, :) - I(1:N-1, :);
DDT = D * D’;
x = y;                                                 % Initialization
Dx = D*x;
Dy = D*y;

for k = 1:Nit
    F = sparse(1:N-1, 1:N-1, abs(Dx)/lam) + DDT;       % F : Sparse banded matrix
    x = y - D’*(F\Dy);                                 % Solve banded linear system
    Dx = D*x;
    cost(k) = 0.5*sum(abs(x-y).^2) + lam*sum(abs(Dx)); % cost function value
end"""

if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/TV-denoise"

    picture = "Shoot-Sienna-Hayes-0042-_DSC0284-sans-PHOTOSHOP-WEB.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # deblur_module(pic, "test-v6", "auto", 19, 0.005, 0, 3,
        #               #mask=[318, 357 + 800, 357, 357 + 440],
        #               refine=True,
        #               backvsmask_ratio=0,
        #               debug=True,
        #               iterations_damping=1.2,
        #               amplify_factor=1
        #               )
        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # deblur_module(pic, "test-v6", "auto", 0, 0.005, 5, 33,
        # mask=[1143, 1143 + 512, 3338, 3338 + 512],
        #              refine=False,
        # backvsmask_ratio=0,
        # denoise=True,
        # debug=True)

        # image, lambd = auto_denoise(pic, 0.6, 0.001)
        pass
