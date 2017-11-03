mport
scipy
from scipy.signal import convolve

import numpy as np
cimport numpy as np

cimport

numpy as np
import numpy as np
import scipy
from cython.parallel import parallel, prange
from scipy.signal import convolve

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack

except:
    pass

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef best_epsilon(image, lambd, p=0.5):
    """
    Find the minimum acceptable epsilon to avoid a degenerate constant solution

    Reference : http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf
    :param image:
    :param lambd:
    :param p:
    :return:
    """
    grad_image = np.gradient(image, edge_order=2)
    norm_grad_image = np.sqrt(grad_image[0] ** 2 + grad_image[1] ** 2)
    omega = 2 * lambd * np.amax(image - image.mean()) / (p * image.size)
    epsilon = np.sqrt(norm_grad_image.mean() / (np.exp(omega) - 1))
    return np.maximum(epsilon, 1e-31)

cdef center_diff(np.ndarray[DTYPE_t, ndim=2] u, int dx, int dy, DTYPE_t epsilon, DTYPE_t p):
    # Centered local difference
    cdef np.ndarray ux, uy, TV, du
    ux = np.roll(u, (dx, 0), axis=(1, 0)) - np.roll(u, (0, 0), axis=(1, 0))
    uy = np.roll(u, (0, dy)) - np.roll(u, (0, 0))
    TV = (np.abs(ux) ** p + np.abs(uy) ** p + epsilon) ** (1 / p)
    du = - ux - uy

    return [TV, du]

cdef x_diff(np.ndarray[DTYPE_t, ndim=2] u, int dx, int dy, DTYPE_t epsilon, DTYPE_t p):
    # x-shifted local difference
    cdef np.ndarray ux, uy, TV, du
    ux = np.roll(u, (0, 0), axis=(1, 0)) - np.roll(u, (-dx, 0), axis=(1, 0))
    uy = np.roll(u, (-dx, dy), axis=(1, 0)) - np.roll(u, (-dx, 0), axis=(1, 0))
    TV = (np.abs(ux) ** p + np.abs(uy) ** p + epsilon) ** (1 / p)
    du = ux

    return [TV, du]

cdef y_diff(np.ndarray[DTYPE_t, ndim=2] u, int dx, int dy, DTYPE_t epsilon, DTYPE_t p):
    # y shifted local difference
    cdef np.ndarray ux, uy, TV, du
    ux = np.roll(u, (dx, -dy), axis=(1, 0)) - np.roll(u, (0, -dy), axis=(1, 0))
    uy = np.roll(u, (0, 0), axis=(1, 0)) - np.roll(u, (0, -dy), axis=(1, 0))
    TV = (np.abs(ux) ** p + np.abs(uy) ** p + epsilon) ** (1 / p)
    du = uy

    return [TV, du]

cpdef np.ndarray[DTYPE_t, ndim=2] gradTVEM(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] ut,
                                           DTYPE_t epsilon=1e-3, DTYPE_t tau=1e-1, DTYPE_t p=0.5):
    """Compute the Total Variation norm of the Minimization-Maximization problem

    :param ndarray image: Input array of pixels
    :return: div(Total Variation regularization term) as described in [3]
    :rtype: ndarray

    We use general P-norm instead : https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-153.pdf

    0.5-norm shows better representation of discontinuities : https://link.springer.com/chapter/10.1007/978-3-319-14612-6_10

    """

    # Displacement vectors of the shifted differences
    cdef np.ndarray deltas
    deltas = np.array([[1, 1],
                       [-1, 1],
                       [1, -1],
                       [-1, -1]])

    # 2-axis shifts
    cdef np.ndarray u_copy = np.zeros_like(u)
    cdef np.ndarray shifts
    shifts = np.array([u_copy,  # Centered
                       u_copy,  # x shifted
                       u_copy  # y shifted
                       ])

    # Methods for local differences calculation
    diffs = [center_diff, x_diff, y_diff]

    # Initialization of the outputs
    cdef np.ndarray du = np.array([shifts, shifts, shifts, shifts])
    cdef np.ndarray TV = du.copy()
    cdef np.ndarray TVt = du.copy()

    cdef int dx, dy, step, i

    with nogil, parallel():
        for i in prange(4):
            # for each displacement vector
            for step in prange(3):
                # for each axial shift
                with gil:
                    dx = deltas[i, 0]
                    dy = deltas[i, 1]
                    TV[i, step], du[i, step] = diffs[step](u, dy, dy, epsilon, p)
                    TVt[i, step], void = diffs[step](ut, dx, dy, epsilon, p)

    grad = np.sum(du / TV / (tau + TVt), axis=(0, 1))

    return grad / 4

def divTV(image):
    """Compute the Total Variation norm

    :param ndarray image: Input array of pixels
    :return: div(Total Variation regularization term) as described in [3]
    :rtype: ndarray

    """
    grad = np.zeros_like(image)

    # Forward differences
    # fx = np.roll(image, 1, axis=1) - image
    fx = np.pad(image, ((0, 0), (1, 0)), mode="edge")[:, 1:] - image
    # fy = np.roll(image, 1, axis=0) - image
    fy = np.pad(image, ((1, 0), (0, 0)), mode="edge")[1:, :] - image
    grad += (fx + fy) / np.maximum(1e-3, np.sqrt(fx ** 2 + fy ** 2))

    # Backward x and crossed y differences
    # fx = image - np.roll(image, -1, axis=1)
    fx = np.pad(image, ((0, 0), (0, 1)), mode="edge")[:, :-1] - image
    # fy = np.roll(image, (-1, 1), axis=(0, 1)) - np.roll(image, -1, axis=0)
    fy = np.pad(image, ((0, 1), (1, 0)), mode="edge")[:-1, 1:] - np.pad(image, ((1, 0), (0, 0)), mode="edge")[1:, :]
    grad -= fx / np.maximum(1e-3, np.sqrt(fx ** 2 + fy ** 2))

    # Backward y and crossed x differences
    #fy = image - np.roll(image, -1, axis=0)
    fy = np.pad(image, ((0, 1), (0, 0)), mode="edge")[:-1, :] - image
    #fx = np.roll(image, (1, -1), axis=(0, 1)) - np.roll(image, -1, axis=1)
    fx = np.pad(image, ((1, 0), (0, 1)), mode="edge")[1:, :-1] - np.pad(image, ((0, 0), (0, 1)), mode="edge")[:, 1:]
    grad -= fy / np.maximum(1e-3, np.sqrt(fy ** 2 + fx ** 2))

    return grad.astype(np.float32)

def _normalize_kernel(kern):
    kern[kern < 0] = 0
    kern /= np.sum(kern, axis=(0, 1))
    return kern

def _convolve_image(u, image, psf):
    error = np.ascontiguousarray(convolve(u, psf, "valid"), np.float32)
    error -= image
    return convolve(error, np.rot90(psf, 2), "full")

def _convolve_kernel(u, image, psf):
    error = np.ascontiguousarray(convolve(u, psf, "valid"), np.float32)
    error -= image
    return convolve(np.rot90(u, 2), error, "valid")

def _update_image_PAM(u, image, psf, lambd, epsilon=5e-3):
    gradu, TV = divTV(u)
    gradu /= TV
    gradu *= lambd
    gradu += _convolve_image(u, image, psf)
    weight = epsilon * np.amax(u) / np.amax(np.abs(gradu))
    u -= weight * gradu
    return u

def _loop_update_image_PAM(u, image, psf, lambd, iterations, epsilon):
    for itt in range(iterations):
        u = _update_image_PAM(u, image, psf, lambd, epsilon)
        lambd *= 0.99
    return u, psf

def _update_kernel_PAM(u, image, psf, epsilon):
    grad_psf = _convolve_kernel(u, image, psf)
    weight = epsilon * np.amax(psf) / np.maximum(1e-31, np.amax(np.abs(grad_psf)))
    psf -= weight * grad_psf
    psf = _normalize_kernel(psf)
    return psf

def _loop_update_both_PAM(u, image, psf, lambd, iterations, mask_u, mask_i, epsilon):
    """
    Utility function to launch the actual deconvolution in parallel
    :param u:
    :param image:
    :param psf:
    :param lambd:
    :param iterations:
    :param mask_u:
    :param mask_i:
    :param epsilon:
    :return:
    """
    for itt in range(iterations):
        u = _update_image_PAM(u, image, psf, lambd, epsilon)
        psf = _update_kernel_PAM(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3]],
                                 image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3]], psf, epsilon)
        lambd *= 0.99
    return u, psf

def _update_both_MM(u, image, psf, lambd, iterations, mask_u, mask_i, epsilon):
    """
    Utility function to launch the actual deconvolution in parallel
    :param u:
    :param image:
    :param psf:
    :param lambd:
    :param iterations:
    :param mask_u:
    :param mask_i:
    :param epsilon:
    :return:
    """
    tau = 1e-3
    lambd = 1 / lambd

    k_step = 1e-3
    u_step = 1e-3

    for it in range(iterations):
        ut = u
        for itt in range(5):
            # Image update
            epsilon = best_epsilon(u, lambd) * 1.001
            gradu = lambd * _convolve_image(u, image, psf) + gradTVEM(u, ut, epsilon, tau)
            dt = u_step * (np.amax(u) + 1 / u.size) / np.amax(np.abs(gradu) + 1e-31)
            u -= dt * gradu

            # PSF update
            gradk = _convolve_kernel(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3]],
                                     image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3]], psf)
            alpha = k_step * (np.amax(psf) + 1 / psf.size) / np.amax(np.abs(gradk) + 1e-31)
            psf -= alpha * gradk
            psf = _normalize_kernel(psf)

        lambd *= 1.001

    return u.astype(np.float32), psf

def pad_image(image: np.ndarray, pad: tuple, mode="edge"):
    """
    Pad an 3D image with a free-boundary condition to avoid ringing along the borders after the FFT

    :param image:
    :param pad:
    :param mode:
    :return:
    """
    R = np.pad(image[..., 0], pad, mode=mode)
    G = np.pad(image[..., 1], pad, mode=mode)
    B = np.pad(image[..., 2], pad, mode=mode)
    u = np.dstack((R, G, B))
    return np.ascontiguousarray(u, np.float32)

def unpad_image(image: np.ndarray, pad: tuple):
    return np.ascontiguousarray(image[pad[0]:-pad[0], pad[1]:-pad[1], ...], np.ndarray)

def richardson_lucy_PAM(image: np.ndarray,
                        u: np.ndarray,
                        psf: np.ndarray,
                        lambd: float,
                        iterations: int,
                        epsilon=1e-3,
                        mask=None,
                        blind=True) -> np.ndarray:
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization by Projected Alternating Minimization.
    This is known to give a close-enough sharp image but never give an accurate sharp image.

    Based on Matlab sourcecode of D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2014: http://www.cvg.unibe.ch/dperrone/tvdb/

    :param ndarray image : Input 3 channels image.
    :param ndarray psf : The point spread function.
    :param int iterations : Number of iterations.
    :param float lambd : Lambda parameter of the total Variation regularization
    :param bool blind : Determine if it is a blind deconvolution is launched, thus if the PSF is updated
        between two iterations
    :returns ndarray: deconvoluted image

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cvg.unibe.ch/dperrone/tvdb/perrone2014tv.pdf
    .. [3] http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf
    .. [4] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [5] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    """

    # image dimensions
    MK, NK, CK = psf.shape
    M, N, C = image.shape

    # Verify the input and scream like a virgin
    assert (CK == C), "Dimensions of the PSF and of the image don't match !"
    assert (MK == NK), "The PSF must be square"
    assert (MK >= 3), "The dimensions of the PSF are too small !"
    assert (M > MK and N > NK), "The size of the picture is smaller than the PSF !"
    assert (MK % 2 != 0), "The dimensions of the PSF must be odd !"

    # Prepare the picture for FFT convolution by padding it with pixels that will be removed
    pad = np.floor(MK / 2).astype(int)
    u = pad_image(u, (pad, pad))

    print("working on image :", u.shape)

    # Adjust the coordinates of the masks with the padding dimensions
    if mask == None:
        mask_i = [0, M + 2 * pad, 0, N + 2 * pad]
        mask_u = [0, M + 2 * pad, 0, N + 2 * pad]
    else:
        mask_u = [mask[0] - pad, mask[1] + pad, mask[2] - pad, mask[3] + pad]
        mask_i = mask

    if blind:
        # Blind deconvolution with PSF refinement
        output = _loop_update_both_PAM,
        [(u[..., 0], image[..., 0], psf[..., 0], lambd, iterations, mask_u, mask_i, epsilon),
         (u[..., 1], image[..., 1], psf[..., 1], lambd, iterations, mask_u, mask_i, epsilon),
         (u[..., 2], image[..., 2], psf[..., 2], lambd, iterations, mask_u, mask_i, epsilon),
         ]
        )

        u = np.dstack((output[0][0], output[1][0], output[2][0]))
        psf = np.dstack((output[0][1], output[1][1], output[2][1]))

        else:
        # Regular deconvolution without PSF refinement
        output = pool.starmap(_loop_update_image_PAM,
        [(u[..., 0], image[..., 0], psf[..., 0], lambd, iterations, epsilon),
         (u[..., 1], image[..., 1], psf[..., 1], lambd, iterations, epsilon),
         (u[..., 2], image[..., 2], psf[..., 2], lambd, iterations, epsilon),
         ]
        )

        u = np.dstack((output[0][0], output[1][0], output[2][0]))
        psf = np.dstack((output[0][1], output[1][1], output[2][1]))

        print(iterations, "iterations")
        u = unpad_image(u, (pad, pad))
        pool.close()

return u.astype(np.float32), psf

def richardson_lucy_MM(image: np.ndarray,
                       u: np.ndarray,
                       psf: np.ndarray,
                       lambd: float,
                       iterations: int,
                       epsilon=5e-3,
                       mask=None,
                       blind=True) -> np.ndarray:
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization by the Minimization-Maximization
    algorithm. This is known to give the sharp image in more than 50 % of the cases.

    Based on Matlab sourcecode of :

    Copyright (C) Daniele Perrone, perrone@iam.unibe.ch
	      Remo Diethelm, remo.diethelm@outlook.com
	      Paolo Favaro, paolo.favaro@iam.unibe.ch
	      2014, All rights reserved.

    :param ndarray image : Input 3 channels image.
    :param ndarray psf : The point spread function.
    :param int iterations : Number of iterations.
    :param float lambd : Lambda parameter of the total Variation regularization
    :param bool blind : Determine if it is a blind deconvolution is launched, thus if the PSF is updated
        between two iterations
    :returns ndarray: deconvoluted image

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cvg.unibe.ch/dperrone/tvdb/perrone2014tv.pdf
    .. [3] http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf
    .. [4] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [5] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    """

    # image dimensions
    MK, NK, C = psf.shape
    M, N, C = image.shape
    pad = np.floor(MK / 2).astype(int)

    print("working on image :", u.shape)

    u = pad_image(u, (pad, pad))

    if mask == None:
        mask_i = [0, M + 2 * pad, 0, N + 2 * pad]
        mask_u = [0, M + 2 * pad, 0, N + 2 * pad]
    else:
        mask_u = [mask[0] - pad, mask[1] + pad, mask[2] - pad, mask[3] + pad]
        mask_i = mask

    iterations = int(np.maximum(iterations / 5, 2))

    cdef int i

    with nogil, parallel():
        for i in prange(30):
            with gil:
                u[..., i], spf[..., i] = _update_both_MM(u[..., i], image[..., i], psf[..., i], lambd, iterations,
                                                         mask_u, mask_i, epsilon)

    u = unpad_image(u, (pad, pad))
    print(iterations * 5, "iterations")
    return u.astype(np.float32), psf
