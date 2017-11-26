'''
Created on 29 oct. 2017

@author: aurelien

'''

import multiprocessing
from os.path import join

import numpy as np
from PIL import Image
from numba import jit, float32, int16
from skimage.restoration import denoise_tv_chambolle

import richardson_lucy_deconvolution as rl
from lib import utils


def auto_denoise(image, k, b):
    """
    Implement an automatic estimation of the best TV-denoising lambda parameter based on the technique of
    Baoxian Wang, Baojun Zhao, Chenwei Deng and Linbo Tang, 2012
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6572468

    https://link.springer.com/article/10.1007/s10915-017-0597-2

    :param image:
    :return:
    """
    # Normalized for 8 bits JPGS
    image = np.ascontiguousarray(image, np.float32) / 255
    TV = rl.divTV(image)
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


@jit(float32[:](float32[:], float32, int16), cache=True, nogil=True)
def _denoise_MM(image, lambd, iterations):
    # http://ai2-s2-pdfs.s3.amazonaws.com/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    tau = 1e-4
    u = image.copy()

    for it in range(iterations):
        ut = u.copy()
        eps = rl.best_param(u)

        for itt in range(5):
            # Image update
            gradu = lambd / 2 * np.abs(image - u) + rl.gradTVEM(u, ut, eps, tau)
            dt = 1e-3 * (np.amax(u)) / np.amax(np.abs(gradu) + 1e-31)
            u -= gradu * dt
            np.clip(u, 0, 1, out=u)

    return image


@utils.timeit
@jit(cache=True)
def denoise_module(pic: np.ndarray, filename: str, dest_path: str, lambd, iterations, effect_strength=1):
    # Backup ICC color profile
    icc_profile = pic.info.get("icc_profile")

    # Assuming 8 bits input, we rescale the RGB values betweem 0 and 1
    image = np.ascontiguousarray(pic, np.float32) / 255

    pool = multiprocessing.Pool(processes=3)
    output = pool.starmap(_denoise_MM,
                          [(image[..., 0], lambd, iterations),
                           (image[..., 1], lambd, iterations),
                           (image[..., 2], lambd, iterations),
                           ]
                          )

    u = np.dstack((output[0], output[1], output[2]))
    pool.close()

    # Convert back into 8 bits RGB
    u = (image - effect_strength * (image - u)) * 255

    utils.save(u, filename, dest_path, icc_profile=icc_profile)


if __name__ == '__main__':
    source_path = "img"
    dest_path = "img/TV-denoise"

    picture = "Shoot-Sienna-Hayes-0042-_DSC0284-sans-PHOTOSHOP-WEB.jpg"
    with Image.open(join(source_path, picture)) as pic:
        # denoise_module(pic, picture + "test-v7-noise", dest_path, 0.005, 100000)
        pass

    picture = "DSC1168.jpg"
    with Image.open(join(source_path, picture)) as pic:
        denoise_module(pic, picture + "test-v7-noise", dest_path, 0.00001, 100000)
        pass
