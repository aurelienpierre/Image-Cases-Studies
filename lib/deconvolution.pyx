# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray
import matplotlib.pyplot as plt
import multiprocessing
from scipy.signal import convolve


cdef int CPU = int(multiprocessing.cpu_count())

cdef extern from "math.h" nogil:
    float fabsf(float)
    float powf(float, float)
    float expf(float)
    float logf(float)
    int isnan(float)
    float atan2f(float, float)
    float cosf(float)
    float sinf(float)


cdef float PI = 3.141592653589793

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cdef inline float gaussian_weight(float source, float target, float sigma) nogil:
    return expf(-powf(source - target, 2) / (2 * powf(sigma, 2) ) ) / (sigma * powf(2 * PI, 0.5))


cdef void gaussian_serie(float[:] serie, float average, float std, int length):
    cdef:
        Py_ssize_t i

    for i in range(length):
        serie[i] = gaussian_weight(serie[i], average, std)


cdef inline void _normalize_kernel(float[:, :, :] kern, int MK) nogil:
    """Normalizes a 3D kernel along its 2 first dimensions"""

    cdef int i, j, k
    cdef float check = 0
    cdef float temp[3]
    temp[0] = 0.
    temp[1] = 0.
    temp[2] = 0.

    # Make negative values == 0
    for k in range(3):
        for i in range(MK):
            for j in range(MK):
                if (kern[i, j, k] < 0):
                    kern[i, j, k] = 0

                temp[k] += kern[i, j, k]

    # Make the sum of elements along the 2 first dimensions == 1
    for k in range(3):
        for i in range(MK):
            for j in range(MK):
                kern[i, j, k] /= temp[k]


cpdef void normalize_kernel(np.ndarray[DTYPE_t, ndim=3] kern, int MK):
    # Expose the Cython function to Python
    _normalize_kernel(kern, MK)


cdef void convolvebis(float[:, :, :] image, float[:, :, :] kernel, float[:, :, :] out, int m, int n, int K) nogil:

    cdef float Sum = 0.

    cdef:
        int i, j, k, ii, jj

    cdef int top = K / 2
    cdef int bottom = m - K / 2 + 1
    cdef int left =  K / 2
    cdef int right = n - K / 2 + 1

    with parallel(num_threads=CPU):
        for i in prange(top, bottom):
            for j in range(left, right):
                for k in range(3):

                    out[i, j, k] = 0.

                    for ii in range(K):
                        for jj in range(K):
                            out[i, j, k]  += image[i + ii - top, j + jj - left, k] * kernel[ii, jj, k]


cdef np.ndarray[DTYPE_t, ndim=3] fft_slice(np.ndarray[DTYPE_t, ndim=2] array, int Ma, int Na, int Mb, int Nb, int domain):
    """Slice the output of a FFT convolution"""

    # FFT convolution input sizes
    cdef Mfft = Ma + Mb - 1
    cdef Nfft = Na + Nb - 1

    # FFT convolution output sizes
    cdef int X, Y

    if domain == 0: # valid
        Y = Ma - Mb + 1
        X = Na - Nb + 1
    elif domain == 1: # full
        Y = Mfft
        X = Nfft
    elif domain == 2: # same
        Y = Ma
        X = Na

    # Offsets
    cdef int offset_Y = int(np.floor((Mfft - Y)/2))
    cdef int offset_X = int(np.floor((Nfft - X)/2))

    return array[offset_Y:offset_Y + Y, offset_X:offset_X + X]


cdef inline float norm_L2(float x, float y, float epsilon) nogil:
    return powf(powf(x, 2) + powf(y, 2) + powf(epsilon, 2), 0.5)


cdef inline float norm_L1(float x, float y, float epsilon) nogil:
    return fabsf(x) + fabsf(y) + epsilon


cdef inline void TV(float[:, :, :] u, float[:, :, :] out, int M, int N, float epsilon, int order, int norm, float[:, :, :] div) nogil:

    cdef:
        Py_ssize_t i, j, k
        float dxdy, adjust
        float udx_forw, udx_back, udy_forw, udy_back, udxdy_back, udxdy_forw, udydx_back, udydx_forw
        float udx, udy, udxdy, udydx

    # Distance over the diagonal
    dxdy = powf(2, 0.5)

    # Normalization coef
    if norm == 1.:
        adjust = 4. * (1 + 1/dxdy)
    elif norm == 2.:
        adjust = 2. * (1 + dxdy)

    ## In this section we treat only the inside of the picture. See the next section for edges and boundaries exceptions

    if order == 2:
        if norm == 1:
            with parallel(num_threads=CPU):
                for i in prange(1, M-1):
                    for j in range(1, N-1):
                        for k in range(3):
                            udx = -2 * u[i, j, k] + u[i-1, j, k] + u[i+1, j, k]
                            udy = -2 * u[i, j, k] + u[i, j-1, k] + u[i, j+1, k]

                            udxdy = (-2 * u[i, j, k] + u[i-1, j-1, k] + u[i+1, j+1, k])/(dxdy)
                            udydx = (-2 * u[i, j, k] + u[i-1, j+1, k] + u[i+1, j-1, k])/(dxdy)

                            div[i, j, k] = -udx - udy - udxdy - udydx
                            div[i, j, k] /= adjust

                            out[i, j, k] = norm_L1(udx, udy, epsilon) + norm_L1(udxdy, udydx, epsilon)
                            out[i, j, k] /= adjust

        else:
            with parallel(num_threads=CPU):
                for i in prange(1, M-1):
                    for j in range(1, N-1):
                        for k in range(3):
                            udx = -2 * u[i, j, k] + u[i-1, j, k] + u[i+1, j, k]
                            udy = -2 * u[i, j, k] + u[i, j-1, k] + u[i, j+1, k]

                            udxdy = (-2 * u[i, j, k] + u[i-1, j-1, k] + u[i+1, j+1, k])/(dxdy)
                            udydx = (-2 * u[i, j, k] + u[i-1, j+1, k] + u[i+1, j-1, k])/(dxdy)

                            div[i, j, k] = -udx - udy - udxdy - udydx
                            div[i, j, k] /= adjust

                            out[i, j, k] = norm_L2(udx, udy, epsilon) + norm_L2(udxdy, udydx, epsilon)
                            out[i, j, k] /= adjust

    elif order == 1:
        if norm == 1:
            with parallel(num_threads=CPU):
                for i in prange(1, M-1):
                    for j in range(1, N-1):
                        for k in range(3):
                            udx_back = u[i, j, k] - u[i-1, j, k]
                            udy_back = u[i, j, k] - u[i, j-1, k]

                            udx_forw = -u[i, j, k] + u[i+1, j, k]
                            udy_forw = -u[i, j, k] + u[i, j+1, k]

                            udxdy_back = (u[i, j, k] - u[i-1, j-1, k])/(dxdy)
                            udydx_back = (u[i, j, k] - u[i-1, j+1, k])/(dxdy)

                            udydx_forw = (-u[i, j, k] + u[i+1, j-1, k])/(dxdy)
                            udxdy_forw = (-u[i, j, k] + u[i+1, j+1, k])/(dxdy)

                            div[i, j, k] = udx_back + udy_back - udx_forw - udy_forw + udxdy_back + udydx_back - udxdy_forw - udydx_forw
                            div[i, j, k] /= adjust

                            out[i, j, k] = norm_L1(udx_back, udy_back, epsilon) + norm_L1(udx_forw, udy_forw, epsilon) + norm_L1(udxdy_back, udydx_back, epsilon) + norm_L1(udxdy_forw, udydx_forw, epsilon)
                            out[i, j, k] /= adjust

        else:
            with parallel(num_threads=CPU):
                for i in prange(1, M-1):
                    for j in range(1, N-1):
                        for k in range(3):
                            udx_back = u[i, j, k] - u[i-1, j, k]
                            udy_back = u[i, j, k] - u[i, j-1, k]

                            udx_forw = -u[i, j, k] + u[i+1, j, k]
                            udy_forw = -u[i, j, k] + u[i, j+1, k]

                            udxdy_back = (u[i, j, k] - u[i-1, j-1, k])/(dxdy)
                            udydx_back = (u[i, j, k] - u[i-1, j+1, k])/(dxdy)

                            udydx_forw = (-u[i, j, k] + u[i+1, j-1, k])/(dxdy)
                            udxdy_forw = (-u[i, j, k] + u[i+1, j+1, k])/(dxdy)

                            div[i, j, k] = udx_back + udy_back - udx_forw - udy_forw + udxdy_back + udydx_back - udxdy_forw - udydx_forw
                            div[i, j, k] /= adjust


                            out[i, j, k] = norm_L2(udx_back, udy_back, epsilon) + norm_L2(udx_forw, udy_forw, epsilon) + norm_L2(udxdy_back, udydx_back, epsilon) + norm_L2(udxdy_forw, udydx_forw, epsilon)
                            out[i, j, k] /= adjust

    ## Warning : borders are ignored !!!


cdef inline void rotate_180(float[:, :, :] array, int M, int N, float[:, :, :] out) nogil:
    """Rotate an array by 2×90° around its center"""

    cdef:
        Py_ssize_t i, j, k

    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    out[i, N-1 -j, k] = array[M - i - 1, j, k]


cdef inline long sign(float trial) nogil:
    cdef long out

    if trial > 0:
        out = +1
    elif trial < 0:
        out = -1
    else:
        out = 0

    return out


cdef inline float mean(float *array, int M, int N, int K) nogil:
    cdef:
        float out = 0
        Py_ssize_t i

    with parallel(num_threads=CPU):
        for i in prange(M*N*K):
            out += array[i]

    return out / (M*N*K)


cdef inline float variance(float *array, float mean, int M, int N, int K) nogil:
    cdef:
        float out = 0
        Py_ssize_t i

    with parallel(num_threads=CPU):
        for i in prange(M*N*K):
            out += powf(mean - array[i], 2)

    return out / (M*N*K)


cdef inline float amax(float *array, int M, int N, int K) nogil:
    cdef:
        Py_ssize_t i
        float out = array[0]

    for i in range(M*N*K):
        if array[i] > out:
            out = array[i]

    return out


cdef inline float amaxabs(float *array, int M, int N, int K) nogil:
    cdef:
        Py_ssize_t i
        float out = fabsf(array[0])
        float temp

    for i in range(M*N*K):
        temp = fabsf(array[i])

        if temp > out:
            out = temp

    return out


cdef inline float array_norm_L2(float *array, int M, int N, int K) nogil:
    cdef:
        Py_ssize_t i
        float out = 0

    for i in range(M*N*K):
        out += powf(array[i], 2)

    return powf(out, 0.5)


cdef inline float array_norm_L1(float *array, int M, int N, int K) nogil:
    cdef:
        Py_ssize_t i
        float out = 0

    for i in range(M*N*K):
        out += fabsf(array[i])

    return out


cpdef np.ndarray[DTYPE_t, ndim=3] richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf, int top, int bottom, int left, int right,
                              float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, int blind=True, int correlation=False, float p=1., int norm=1, int order=2, float priority=0, int refocus=0):
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
    .. [6] http://awibisono.github.io/2016/06/20/accelerated-gradient-descent.html
    .. [7] https://www.sciencedirect.com/science/article/pii/S0307904X13001832
    """

    cdef int u_M = u.shape[0]
    cdef int u_N = u.shape[1]
    cdef Py_ssize_t it, itt, i, j, k
    cdef int inner_iter = 5
    cdef int pad = (u_M - M)//2

    cdef np.ndarray[DTYPE_t, ndim=3] gradk = np.empty((MK, MK, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] ut = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] utemp = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] gradu = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] synth = np.zeros((M, N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] error = np.zeros((M, N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] div = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] div_ut = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] TV_ut_L1 = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] TV_ut_L2 = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] TV_u_L2 = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] TV_u_L1 = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] DoF = np.zeros((M, N, 3), dtype=DTYPE)

    # Construct the array of gaussian weights for the autocovariance metric
    cdef np.ndarray[DTYPE_t, ndim=2] weights = np.zeros((bottom - top, right - left), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] width, height
    cdef np.ndarray[DTYPE_t, ndim=3] test = np.zeros((bottom-top, right-left, 3), dtype=DTYPE)

    width = np.linspace(-1., 1., num=(bottom - top), dtype=DTYPE)
    height = np.linspace(-1., 1., num=(right - left), dtype=DTYPE)

    gaussian_serie(width, 0., 1., bottom - top)
    gaussian_serie(height, 0., 1., right - left)

    weights = np.sqrt(np.outer(width, height))
    weights /= np.sum(weights)

    cdef float dt[3]
    cdef float dtpsf
    cdef float max_error[3]
    cdef float max_float
    cdef float M_r, M_r_prev, min_M_r, temp, alpha, beta, v, lambdt, lambdtt
    cdef float logDu, logDut, varu, varut, Hu, Hut, increase
    cdef float dvar = 0.
    cdef float dH = 0.
    cdef float sigma = 1.

    # Once convergence is detected or error goes out of bounds, the stop_flag is raised
    cdef int stop_flag = False
    cdef int p_flag = False
    cdef int convergence_flag = False

    # Temporary buffers for intermediate computations - we declare them here to avoid Python calls in sub-routines
    cdef float[:, :, :] psf_rotated = cvarray(shape=(MK, MK, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] u_rotated = cvarray(shape=(u_M, u_N, 3), itemsize=sizeof(float), format="f")


    # Estimate the noise and the parameters
    cdef float B, H, B_previous, H_previous, B_first
    cdef float epsilon

    # Epsilon is the parameter of the Cauchy distribution of the gradients along the image
    # In a blind setup, given that the PSF estimation window is narrow and consistently chosen
    # epsilon is big, meaning the gradients distribution has a heavy tail
    # In a non-blind setup, epsilon has lower bound bounded by the ratio between
    if blind:
        epsilon = 1e-2
    else:
        epsilon = 1e-6#(bottom - top) * (right-left) / (M*N)

    cdef float conv = 0.125

    rotate_180(psf, MK, MK, psf_rotated)

    # Compute alpha & beta
    # Priority in [-1, 1] : negative gives more power to denoising, positive to deblurring, 0 is even

    if priority > 0:
        beta = (abs(priority)+1)/2
        alpha = 1 - beta
    elif priority < 0:
        alpha = (abs(priority)+1)/2
        beta = 1 - alpha
    elif priority == 0:
        alpha = 0.5
        beta = 0.5

    it = 0


    ## From Perrone & Favaro : A logaritmic prior for blind deconvolution
    while it < iterations and not stop_flag:
        # Majorization loop
        ut[:] = u.copy()
        # Compute the ratio of the epsilon-norm of Total Variation between the current major and minor deblured images
        #TV(ut, TV_u_L1, u_M, u_N, epsilon, 2, 1, div_ut)
        #TV(ut, TV_u_L2, u_M, u_N, epsilon, 2, 2, div_ut)

        itt = 0

        if it > 0:
            B_previous = B
            H_previous = H

        while itt < inner_iter:
            ## Deblurring minimization loop

            # Sythesize the blur
            for chan in range(3):
                synth[..., chan] = convolve(u[..., chan], psf[..., chan], mode="valid")

            #convolvebis(u, psf, error, M, N, MK)

            with nogil:
                # Compute the residual
                with parallel(num_threads=CPU):
                    for i in prange(M):
                        for j in range(N):
                            for k in range(3):
                                error[i, j, k] =  synth[i, j, k] - image[i, j, k]

            for k in range(3):
                gradu[..., k] = convolve(error[..., k], psf_rotated[..., k], mode="full")


            # Compute the ratio of the epsilon-norm of Total Variation between the current major and minor deblured images
            TV(u, TV_u_L1, u_M, u_N, epsilon, 2, 1, div)
            TV(u, TV_u_L2, u_M, u_N, epsilon, 2, 2, div)

            # Compute the depth of field mask by measuring the error between the estimated blur and the denoised blurry image
            DoF = ((gradu[pad:-pad, pad:-pad, ...] - image) / (gradu[pad:-pad, pad:-pad, ...] + image))**2
            #DoF /= np.amax(DoF)
            if not blind:
              DoF /= lambd

            #TODO : use the hyper-laplacian prior for non-blind deblurring
            #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.154.539&rep=rep1&type=pdf

            # Regularization step
            with nogil:
                ## Adapted from Perrone & Favaro with Lv, Song, Wang & Le : Image restoration with a high-order total variation minimization method
                ## https://ac.els-cdn.com/S0307904X13001832/1-s2.0-S0307904X13001832-main.pdf?_tid=05b19487-7e64-4d36-823e-3391f48b6e6a&acdnat=1530684412_d946ea0d68205a7991c34b3ad60b0dd7
                # Second order TV minimization problem
                with parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for j in range(u_N):
                             for k in range(3):
                                if TV_ut_L1[i, j, k] != 0 and TV_u_L1[i, j, k] != 0:
                                    gradu[i, j, k] = div[i, j, k] / TV_u_L1[i, j, k] / TV_ut_L1[i, j, k] / 2. + div[i, j, k] / TV_u_L2[i, j, k] / TV_ut_L2[i, j, k] / 2. + lambd * gradu[i, j, k] + (u[i, j, k] - ut[i, j, k])/4. #+ div_ut[i, j, k] / TV_ut_L2[i, j, k] / TV_u_L2[i, j, k] + div_ut[i, j, k] / TV_ut_L1[i, j, k] / TV_u_L1[i, j, k]
                                else:
                                    gradu[i, j, k] = lambd * gradu[i, j, k] + (u[i, j, k] - ut[i, j, k]) / 2.


            # Scale the gradu factor
            for k in range(3):
                dt[k] = step_factor * (np.amax(u[..., k]) + 1/(u_M * u_N)) / (np.amax(np.abs(gradu[..., k])) + 1e-15)

            # Update the deblurred picture
            with nogil, parallel(num_threads=CPU):
                for i in prange(u_M):
                    for j in range(u_N):
                         for k in range(3):
                            u[i, j, k] -= dt[k] * gradu[i, j, k]

            # Denoise the original blurry image to improve the comparison with the synthesized blur
            with nogil:
                ## Adapted from Perrone & Favaro with Lv, Song, Wang & Le : Image restoration with a high-order total variation minimization method
                ## https://ac.els-cdn.com/S0307904X13001832/1-s2.0-S0307904X13001832-main.pdf?_tid=05b19487-7e64-4d36-823e-3391f48b6e6a&acdnat=1530684412_d946ea0d68205a7991c34b3ad60b0dd7
                # Second order TV minimization problem
                with parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for j in range(u_N):
                             for k in range(3):
                                if TV_ut_L1[i, j, k] != 0 and TV_u_L1[i, j, k] != 0:
                                    gradu[i, j, k] = div[i, j, k] / TV_u_L1[i, j, k] / TV_ut_L1[i, j, k] / 2. + div[i, j, k] / TV_u_L2[i, j, k] / TV_ut_L2[i, j, k] / 2.
                                else:
                                    gradu[i, j, k] = 0.

            for k in range(3):
                dt[k] = step_factor * (np.amax(image[..., k]) + 1/(M * N)) / (np.amax(np.abs(gradu[..., k])) + 1e-15)
                image[..., k] -= dt[k] * gradu[pad:-pad, pad:-pad, k] / lambd

            # Retain some of the blurry image in the zones where no satisfaying deblurring has been performed
            u[pad:-pad, pad:-pad, ...] = (1. - DoF) * u[pad:-pad, pad:-pad, ...] + DoF * image

            # PSF update
            if blind and not stop_flag:
                # Synthesize the blur
                for chan in range(C):
                    error[..., chan] = convolve(u[..., chan], psf[..., chan], mode="valid")

                # Compute the residual
                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                error[i, j, k] -= image[i, j, k]

                rotate_180(u, u_M, u_N, u_rotated)

                # Correlate the rotated deblurred image and the residual
                for chan in range(C):
                    gradk[..., chan] = convolve(u_rotated[..., chan], error[..., chan], mode="valid")

                # Scale the gradk factor
                dtpsf = step_factor / MK * (np.amax(psf) + 1/(u_M * u_N * 3)) / (np.amax(np.abs(gradk)) + float(1e-15))

                # Update the PSF
                with nogil, parallel(num_threads=CPU):
                    for i in prange(MK):
                        for k in range(3):
                            for j in range(MK):
                                psf[i, j, k] -= dtpsf * gradk[i, j, k]

                # When unexpected decorrelation happens between channels (patterns and geometric shapes), it might help to bound them
                if correlation:
                    psf = np.dstack((np.mean(psf, axis=2), np.mean(psf, axis=2), np.mean(psf, axis=2)))

                _normalize_kernel(psf, MK)

                rotate_180(psf, MK, MK, psf_rotated)

            itt += 1

        print("DoF : min = %f | max = %f" % (np.amin(DoF), np.amax(DoF)))

        # Compute the TV L2 for lambda estimation
        if it > 0:
            varut = varu
            Hut = Hu

        varu = np.std(u[top+pad:bottom-pad, left+pad:right-pad, ...])**2
        Hu = np.linalg.norm(error[top:bottom, left:right, ...])**2 / ((bottom-top) * (right-left) * 3)
        """
        # Roll back the history
        if it > 1:
            lambdtt = lambdt
        if it > 0:
            lambdt = lambd

        # Update lambda
        if Hu > 0.:
            if it > 0:
                if (varu + varut) > 0. and (Hu + Hut) > 0.:

                    dvar = (varu - varut)/(varu + varut)
                    dH = (Hu - Hut)/(Hu + Hut)

        if it > 0:
            print("%.5E, %.4E, %.6E, %.6E, %.3E," % (lambd, epsilon, varu , Hu, p))
        """
        ### Convergence analysis
        ## From Almeida & Figueiredo : New stopping criteria for iterative blind image deblurring based on residual whiteness measures
        ## http://www.lx.it.pt/~mtf/Almeida_Figueiredo_SSP2011.pdf
        if it > 0:
            M_r_prev = M_r

        # Center the mean at zero
        test = (error[top:bottom, left:right, ...] - np.mean(error[top:bottom, left:right, ...]))/ np.std(error[top:bottom, left:right, ...])
        # Normalize between -1 and 1
        test /= np.amax(np.abs(test))
        # Autocorrelate the picture : autocovariance
        for k in range(3):
            test[..., k] = convolve(test[..., k], np.rot90(test[..., k], 2), mode="same")
            # Compute the white noise metric
            test[..., k] = test[..., k]**2 * weights

        # We are supposed to take the sum here, but then the threshold would not be size-invariant
        # The mean is supposed to give the same number no matter the size of the patch
        M_r = np.mean(test)

        if it == inner_iter:
            min_M_r = M_r

        if it > 1:
            if blind:
                # Get more conservative on the error tolerance if the PSF is being estimated
                if M_r > M_r_prev:
                    stop_flag = True
                    print("white autocorellation condition met")

            else:
                # Get more sloppy on the error if we do a non-blind deconvolution
                if (M_r - M_r_prev) / (M_r + M_r_prev)  > tau:# or M_r > (1 + tau) * min_M_r:
                    stop_flag = True
                    print("white autocorellation condition met")

        it += 1

        if it % 50 == 0:
            print("%i iterations completed" % (it))

    if stop_flag:
        # When one stopping condition has been met, the solutions u and psf have already past degeneration by one step
        # So we retrieve and output the solution from the step before

        print("Convergence after %i iterations." % (it))
    else:
        print("Did not converge after %i iterations. Don't use the result." % it)

    print("Stats : autocovariance = %.6f | lamdba = %.0f | residual = %.6f | variance/noise = %.6f" % (1000 * M_r/((bottom - top)*(right - left)*3), lambd, Hu, varu))

    if np.any(np.isnan(u)):
      print("has NaN after DoF correction")

    # Return u where image is defined to not output some border effects
    return u[pad:pad + M, pad:pad + N, ...]
