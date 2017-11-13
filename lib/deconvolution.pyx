# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division


import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange
import scipy.signal
import multiprocessing


cdef int CPU = multiprocessing.cpu_count()


try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack
    a = np.empty((900, 600))
    print("Profiling the system for performance…")
    pyfftw.builders.rfft2(a, s=(900, 600), overwrite_input=True, planner_effort='FFTW_MEASURE', threads=CPU,
                          auto_align_input=True, auto_contiguous=True, avoid_copy=False)
    print("Profiling done !")
except:
    pass


cdef extern from "stdlib.h":
    float abs(float x) nogil


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cdef void shift(float[:, :] u, float[:, :] us, int dy, int dx) nogil:
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]

    cdef int i_min = max([-dy, 0])
    cdef int i_max = min([M, M-dy])

    cdef int j_min = max([-dx, 0])
    cdef int j_max = min([N, N-dx])

    cdef int i_min_1 = max([dy, 0])
    cdef int i_max_1 = min([dy+M, M])

    cdef int j_min_1 = max([dx, 0])
    cdef int j_max_1 = min([dx+N, N])

    cdef float[:, :] begin = u[i_min_1:i_max_1, j_min_1:j_max_1]
    cdef float[:, :] end = u[i_min:i_max, j_min:j_max]
    cdef float[:, :] diff = us[i_min:i_max, j_min:j_max]

    M = diff.shape[0]
    N = diff.shape[1]

    cdef int i, j

    with parallel(num_threads=8):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                diff[i, j] = begin[i, j] - end[i, j]


cdef void TV_norm_p(float[:, :] ux, float[:, :] uy, float[:, :] output, float epsilon, float p) nogil:
    cdef int M = ux.shape[0]
    cdef int N = ux.shape[1]
    cdef int i, j
    cdef float inv_p = 1./p
    cdef float eps = epsilon**p

    with parallel(num_threads=8):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                output[i, j] += (abs(ux[i, j])** p + abs(uy[i, j])** p + eps) **inv_p


cdef void TV_norm_one(float[:, :] ux, float[:, :] uy, float[:, :] output, float epsilon) nogil:
    cdef int M = ux.shape[0]
    cdef int N = ux.shape[1]
    cdef int i, j

    with parallel(num_threads=8):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                output[i, j] += abs(ux[i, j]) + abs(uy[i, j]) + epsilon


cdef void TV_norm(float[:, :] ux, float[:, :] uy, float[:, :] output, float epsilon, float p) nogil:
    if p == 1:
        TV_norm_one(ux, uy, output, epsilon)
    else:
        TV_norm_p(ux, uy, output, epsilon, p)


cpdef void center_diff(float[:, :] u, float[:, :] TV, int di, int dj, float epsilon, float p):
    # Centered local difference
    cdef float[:, :]  ux =  np.zeros_like(u)
    cdef float[:, :]  uy =  np.zeros_like(u)
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=8):

        shift(u, ux, di, 0)
        shift(u, uy, 0, dj)

        for i in prange(M, schedule="guided"):
            for j in range(N):
                ux[i, j] -= u[i, j]

        for i in prange(M, schedule="guided"):
            for j in range(N):
                uy[i, j] -= u[i, j]

        # for i in prange(M, schedule="guided"):
        #     for j in range(N):
        #         du[i, j] -= ux[i, j] + uy[i, j]

        TV_norm(ux, uy, TV, epsilon, p)


cpdef void i_diff(float[:, :]  u, float[:, :] TV, int di, int dj, float epsilon, float p):
    # line-shifted local difference
    cdef float[:, :]  ux =  np.zeros_like(u)
    cdef float[:, :]  uy =  np.zeros_like(u)
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=8):

        shift(u, ux, -di, 0)
        shift(u, uy, -di, dj)

        for i in prange(M, schedule="guided"):
            for j in range(N):
                uy[i, j] -= ux[i, j]

        for i in prange(M, schedule="guided"):
            for j in range(N):
                ux[i, j] = u[i, j] - ux[i, j]

        # for i in prange(M, schedule="guided"):
        #     for j in range(N):
        #         du[i, j] += ux[i, j]

        TV_norm(ux, uy, TV, epsilon, p)


cpdef void j_diff(float[:, :]  u, float[:, :] TV, int di, int dj, float epsilon, float p):
    # column-shifted local difference
    cdef float[:, :]  ux =  np.zeros_like(u)
    cdef float[:, :]  uy =  np.zeros_like(u)
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=8):

        shift(u, uy, 0, -dj)
        shift(u, ux, di, -dj)

        for i in prange(M, schedule="guided"):
            for j in range(N):
                ux[i, j] -= uy[i, j]

        for i in prange(M, schedule="guided"):
            for j in range(N):
                uy[i, j] = u[i, j] - uy[i, j]

        # for i in prange(M, schedule="guided"):
        #     for j in range(N):
        #         du[i, j] += uy[i, j]

        TV_norm(ux, uy, TV, epsilon, p)


cdef void divTV(float[:, :] u, float[:, :] TV, float epsilon=0, float p=1):
    # cdef float[:, :] TV = np.zeros_like(u)
    # cdef float[:, :] du = np.zeros_like(u)
    cdef int i, j, di, dj
    cdef list shifts = [[1, 1, center_diff],
                    [-1, 1, center_diff],
                    [1,-1, center_diff],
                    [-1, -1, center_diff],
                    [1, 1, i_diff],
                    [-1, 1, i_diff],
                    [1,-1, i_diff],
                    [-1, -1, i_diff],
                    [1, 1, j_diff],
                    [-1, 1, j_diff],
                    [1,-1, j_diff],
                    [-1, -1, j_diff]
                   ]

    with nogil, parallel(num_threads=8):
        for i in prange(12, schedule="guided"):
            with gil:
                di = shifts[i][0]
                dj = shifts[i][1]
                method = shifts[i][2]
                method(u, TV, di, dj, epsilon, p)

    # return TV


cdef float[:, :] gradTVEM(float[:, :] u, float[:, :] ut, float epsilon=1e-3, float tau=1e-3, float p=1):
    cdef float[:, :] out
    cdef float[:, :] TV = np.zeros_like(u)
    # cdef float[:, :] TV, du
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    divTV(u, TV, epsilon=epsilon, p=p)
    out = TV

    # with nogil, parallel(num_threads=64):
    #     for i in prange(M, schedule="guided"):
    #         for j in range(N):
    #             out[i, j] += du[i, j] / TV[i, j]

    divTV(ut, TV, epsilon=epsilon, p=p)

    with nogil, parallel(num_threads=8):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                out[i, j] /= (TV[i, j] + tau)
                # out[i, j] /= 4


    return out


cdef float best_param(np.ndarray[DTYPE_t, ndim=2] image, float lambd, float p=1):
    """
    Determine by a statistical method the best lambda parameter. [1]

    Find the minimum acceptable epsilon to avoid a degenerate constant solution [2]

    Reference :
    .. [1] https://pdfs.semanticscholar.org/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    .. [2] http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf

    :param image:
    :return:
    """

    cdef list grad = np.gradient(image)
    cdef float grad_std = np.linalg.norm(image - image.mean()) / image.size
    cdef float grad_mean = np.linalg.norm(grad) / image.size

    # lambd = noise_reduction_factor * np.sum(np.sqrt(divTV(image, p=1)))**2 / (-np.log(np.std(image)**2) * * 2 * np.pi)

    cdef float omega = 2 * lambd * grad_std / p
    cdef float epsilon = np.sqrt(grad_mean / (np.exp(omega) - 1))

    # print(lambd, epsilon, p)
    return epsilon * 1.001


cpdef np.ndarray _normalize_kernel(np.ndarray kern):
    """
    This function is not Cythonized since it can take 2D and 3D outputs and works on small kernels
    
    :param kern: 
    :return: 
    """
    # Make the negative values = 0
    kern[kern < 0] = 0
    # Make the sum of the kernel elements = 1
    kern /= np.sum(kern, axis=(0, 1))
    return kern.astype(np.float32)


from pyfftw.builders import rfft2, irfft2

cdef np.ndarray[DTYPE_t, ndim=2] ftconvolve(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, str domain):
    cdef int MK =  b.shape[0]
    cdef int NK = b.shape[1]
    cdef int M = a.shape[0]
    cdef int N = b.shape[1]

    cdef np.ndarray[np.complex64_t, ndim=2] ahat, bhat, chat
    cdef np.ndarray[DTYPE_t, ndim=2] c

    ahat = rfft2(a, s=(M + MK -1, N + NK -1), threads=8).output_array
    bhat = rfft2(a, s=(M + MK -1, N + NK -1), threads=8).output_array
    chat = ahat * bhat

    cdef int Y, X

    if domain =="same":
        Y = M
        X = N
    elif domain == "valid":
        Y = M - MK
        X = N - NK

    print(chat)

    c = (irfft2(chat, s=(Y, X), threads=8).output_array).astype(DTYPE)

    return c

cdef convolve = scipy.signal.fftconvolve

cdef np.ndarray[DTYPE_t, ndim=2] _convolve_image(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf):

    cdef np.ndarray[DTYPE_t, ndim=2] error
    error = convolve(u, psf, "valid").astype(DTYPE)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=8):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                error[i, j] -= image[i, j]

    error = convolve(error, np.rot90(psf, 2), "full").astype(DTYPE)

    return error


cdef np.ndarray[DTYPE_t, ndim=2] _convolve_kernel(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf):

    cdef np.ndarray[DTYPE_t, ndim=2] error
    error = convolve(u, psf, "valid").astype(DTYPE)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=8):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                error[i, j] -= image[i, j]

    error = convolve(np.rot90(u, 2), error, "valid").astype(DTYPE)

    return error


cdef void _update_both_MM(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf,
                    float lambd, int iterations, list mask_u, list mask_i, float epsilon, int blind, float p):

    cdef float k_step = epsilon
    cdef float u_step = epsilon

    cdef np.ndarray[DTYPE_t, ndim=2] u_masked = u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3]]
    cdef np.ndarray[DTYPE_t, ndim=2] im_masked = image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3]]

    cdef np.ndarray[DTYPE_t, ndim=2] ut
    cdef float eps, dt, alpha, max_gradu, abs_gradu

    cdef np.ndarray[DTYPE_t, ndim=2] gradu, gradk, im_convo
    cdef float[:, :] gradV
    gradk = np.zeros_like(psf)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int MK = psf.shape[0]
    cdef int NK = psf.shape[1]
    cdef int i, j

    for it in range(iterations):
        ut = u.copy()
        lambd = min([lambd, 50000])
        eps = best_param(u, lambd, p=p)

        for itt in range(5):
            # Image update
            lambd = min([lambd, 50000])

            gradV = gradTVEM(u, ut, eps, eps, p=p)
            im_convo = _convolve_image(u, image, psf)
            gradu = np.zeros_like(u)
            max_gradu = 0

            with nogil, parallel(num_threads=8):
                for i in prange(M, schedule="guided"):
                    for j in range(N):
                        gradu[i, j] = lambd * im_convo[i, j] + gradV[i, j]

            dt = u_step * (np.amax(u) + 1 / (M*N)) / (np.amax(np.abs(gradu)) + 1e-31)

            with nogil, parallel(num_threads=8):
                for i in prange(M, schedule="guided"):
                    for j in range(N):
                        u[i, j] -= dt * gradu[i, j]

                        # Clipping of out-of-range values
                        if u[i, j] > 1:
                            u[i, j] = 1

                        elif u[i, j] < 0:
                            u[i, j] = 0

            if blind:
                # PSF update
                gradk = _convolve_kernel(u_masked, im_masked, psf)
                alpha = k_step * (np.amax(psf) + 1 / (MK*NK)) / np.amax(np.abs(gradk) + 1e-31)
                psf -= alpha * gradk
                psf = _normalize_kernel(psf)

            lambd *= 1.001

        print("%i/%i iterations completed" % (it * 5, iterations*5))

cpdef pad_image(image, pad, mode="edge"):
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


cpdef unpad_image(image, pad):
    return np.ascontiguousarray(image[pad[0]:-pad[0], pad[1]:-pad[1], ...], np.ndarray)


cdef list _richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                              float lambd, int iterations, float epsilon, list mask=None, int blind=True, float p=1):
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
    cdef int MK = psf.shape[0]
    cdef int NK = psf.shape[1]

    cdef M = image.shape[0]
    cdef N = image.shape[1]

    cdef int pad = np.floor(MK / 2).astype(int)

    u = pad_image(u, (pad, pad)).astype(np.float32)

    if mask == None:
        mask_i = [0, M + 2 * pad, 0, N + 2 * pad]
        mask_u = [0, M + 2 * pad, 0, N + 2 * pad]
    else:
        mask_u = [mask[0] - pad, mask[1] + pad, mask[2] - pad, mask[3] + pad]
        mask_i = mask

    cdef int i

    with nogil, parallel(num_threads=3):
        for i in prange(3, schedule="guided"):
            with gil:
                _update_both_MM(u[..., i], image[..., i], psf[..., i], lambd, iterations, mask_u, mask_i, epsilon, blind, p)

    u = u[pad:-pad, pad:-pad, ...]

    return [u.astype(np.float32), psf]

def richardson_lucy_MM(image, u, psf, lambd, iterations, epsilon, mask=None, blind=True, p=1):
    return  _richardson_lucy_MM(image, u, psf, lambd, iterations, epsilon, mask=None, blind=True, p=1)
