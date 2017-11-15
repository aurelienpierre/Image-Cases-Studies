# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=True

from __future__ import division

import numpy as np
cimport numpy as np
import time
cimport cython
from cython.parallel cimport parallel, prange
import multiprocessing
import psutil

#from scipy.signal import convolve
import pyfftw

cdef int CPU = multiprocessing.cpu_count()

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

    with parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                diff[i, j] = begin[i, j] - end[i, j]


cdef void TV_norm_p(float[:, :] ux, float[:, :] uy, float[:, :] output, float epsilon, float p) nogil:
    cdef int M = ux.shape[0]
    cdef int N = ux.shape[1]
    cdef int i, j
    cdef float inv_p = 1./p
    cdef float eps = epsilon**p

    with parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                output[i, j] += (abs(ux[i, j])** p + abs(uy[i, j])** p + eps) **inv_p


cdef void TV_norm_one(float[:, :] ux, float[:, :] uy, float[:, :] output, float epsilon) nogil:
    cdef int M = ux.shape[0]
    cdef int N = ux.shape[1]
    cdef int i, j

    with parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                output[i, j] += abs(ux[i, j]) + abs(uy[i, j]) + epsilon


cdef void TV_norm(float[:, :] ux, float[:, :] uy, float[:, :] output, float epsilon, float p) nogil:
    if p == 1:
        TV_norm_one(ux, uy, output, epsilon)
    else:
        TV_norm_p(ux, uy, output, epsilon, p)


cpdef void center_diff(float[:, :]  u, float[:, :] TV, float[:, :] ux, float[:, :] uy, int di, int dj, float epsilon, float p) nogil:
    # Centered local difference
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    with parallel(num_threads=CPU):

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


cpdef void i_diff(float[:, :]  u, float[:, :] TV, float[:, :] ux, float[:, :] uy, int di, int dj, float epsilon, float p) nogil:
    # line-shifted local difference
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    with parallel(num_threads=CPU):

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


cpdef void j_diff(float[:, :]  u, float[:, :] TV, float[:, :] ux, float[:, :] uy, int di, int dj, float epsilon, float p) nogil:
    # column-shifted local difference
    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int i, j

    with parallel(num_threads=CPU):

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
    cdef np.ndarray[np.int8_t, ndim=2] shifts = np.array([[1, 1], [-1, 1], [1,-1], [-1, -1]], dtype=np.int8)

    cdef float[:, :]  ux =  np.zeros_like(u)
    cdef float[:, :]  uy =  np.zeros_like(u)

    with nogil:
        for i in range(4):
                di = shifts[i, 0]
                dj = shifts[i, 1]
                center_diff(u, TV, ux, uy, di, dj, epsilon, p)
                i_diff(u, TV, ux, uy, di, dj, epsilon, p)
                j_diff(u, TV, ux, uy, di, dj, epsilon, p)

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

    with nogil, parallel(num_threads=CPU):
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
    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef float image_mean = image.mean()
    cdef list grad = np.gradient(image)
    cdef float grad_mean = np.linalg.norm(grad) / (M*N)
    cdef float grad_std = np.linalg.norm(image - image_mean) / (M*N)


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


cdef class convolve:
    cdef object fft_A_obj, fft_B_obj, ifft_obj,  output_array
    cdef int X, Y, M, N
    cdef bint lock

    def __cinit__(self, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B, str domain):
        cdef int MK =  B.shape[0]
        cdef int NK = B.shape[1]
        cdef int M = A.shape[0]
        cdef int N = A.shape[1]

        if domain =="same":
            self.Y = M
            self.X = N
        elif domain == "valid":
            self.Y = M - MK + 1
            self.X = N - NK + 1
        elif domain == "full":
            self.Y = M + MK - 1
            self.X = N + NK - 1

        self.M = M + MK -1
        self.N = N + NK -1

        print("System profiling…")
        self.fft_A_obj = pyfftw.builders.rfft2(A, s=(self.M, self.N), threads=CPU, auto_align_input=True, auto_contiguous=True)
        self.fft_B_obj = pyfftw.builders.rfft2(B, s=(self.M, self.N ), threads=CPU,  auto_align_input=True, auto_contiguous=True)
        self.ifft_obj = pyfftw.builders.irfft2(self.fft_B_obj.output_array , s=(self.Y, self.X), threads=CPU, auto_align_input=True, auto_contiguous=True)
        print("Profiling done !")

        self.output_array = self.ifft_obj.output_array

        self.lock = False

    def __call__(self, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B):
        """
        The lock mechanism makes it thread-safe
        :param A:
        :param B:
        :return:
        """
        cdef np.ndarray[DTYPE_t, ndim=2] R

        if self.lock == False:
            self.lock = True
            R = self.ifft_obj(np.fft.ifftshift(np.fft.fftshift(self.fft_A_obj(A)) * np.fft.fftshift(self.fft_B_obj(B))))
            self.lock = False
        else:
            while self.lock == True:
                time.sleep(0.005)
                if self.lock == False:
                    self.lock = True
                    R = self.ifft_obj(np.fft.ifftshift(np.fft.fftshift(self.fft_A_obj(A)) * np.fft.fftshift(self.fft_B_obj(B))))
                    self.lock = False
                    break


        """
        int m, n;      // FFT row and column dimensions might be different
        int m2, n2;
        int i, k;
        complex x[m][n];
        complex tmp13, tmp24;
        
        m2 = m / 2;    // half of row dimension
        n2 = n / 2;    // half of column dimension
        
        // interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
        
        for (i = 0; i < m2; i++)
        {
             for (k = 0; k < n2; k++)
             {
                  tmp13         = x[i][k];
                  x[i][k]       = x[i+m2][k+n2];
                  x[i+m2][k+n2] = tmp13;
        
                  tmp24         = x[i+m2][k];
                  x[i+m2][k]    = x[i][k+n2];
                  x[i][k+n2]    = tmp24;
             }
        }
                
        """

        return R

from scipy.signal import fftconvolve

#cdef float[:, :] _convolve_image(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf, convolve FFT_valid, convolve FFT_full):
cdef float[:, :] _convolve_image(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf):
    cdef np.ndarray[DTYPE_t, ndim=2] error
    #error = FFT_valid(u, psf)
    error = fftconvolve(u, psf, "valid").astype(DTYPE)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                error[i, j] -= image[i, j]

    #return FFT_full(error, np.rot90(psf, 2))
    return fftconvolve(error, np.rot90(psf, 2), "full").astype(DTYPE)


#cdef np.ndarray[DTYPE_t, ndim=2] _convolve_kernel(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf, convolve FFT_valid, convolve FFT_kern_valid):
cdef np.ndarray[DTYPE_t, ndim=2] _convolve_kernel(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf):
    cdef np.ndarray[DTYPE_t, ndim=2] error

    #error = FFT_valid(u, psf)
    error = fftconvolve(u, psf, "valid").astype(DTYPE)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                error[i, j] -= image[i, j]

    #return FFT_kern_valid(np.rot90(u, 2), error)
    return fftconvolve(np.rot90(u, 2), error, "valid").astype(DTYPE)


cdef void _update_both_MM(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf,
                    float lambd, int iterations, np.ndarray[DTYPE_t, ndim=2] u_masked, np.ndarray[DTYPE_t, ndim=2] im_masked,
                          float epsilon, int blind, float p):#, convolve FFT_valid, convolve FFT_full, convolve FFT_kern_valid):

    cdef float k_step = epsilon
    cdef float u_step = epsilon

    cdef float eps, dt, alpha, max_gradu, abs_gradu

    cdef np.ndarray[DTYPE_t, ndim=2] gradu, gradk
    cdef float[:, :] gradV, im_convo, ut
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
            im_convo = _convolve_image(u, image, psf)#, FFT_valid, FFT_full)
            gradu = np.zeros_like(u)
            max_gradu = 0

            with nogil, parallel(num_threads=CPU):
                for i in prange(M, schedule="guided"):
                    for j in range(N):
                        gradu[i, j] = lambd * im_convo[i, j] + gradV[i, j]

            dt = u_step * (np.amax(u) + 1 / (M*N)) / (np.amax(np.abs(gradu)) + 1e-31)

            with nogil, parallel(num_threads=CPU):
                for i in prange(M, schedule="guided"):
                    for j in range(N):
                        u[i, j] -= dt * gradu[i, j]

                        if u[i, j] < 0:
                            u[i, j] = 0
                        if u[i, j] > 1:
                            u[i, j] = 1

            if blind:
                # PSF update
                gradk = _convolve_kernel(u_masked, im_masked, psf)#, FFT_valid, FFT_kern_valid)
                alpha = k_step * (np.amax(psf) + 1 / (MK*NK)) / np.amax(np.abs(gradk) + 1e-31)
                psf -= alpha * gradk
                psf = _normalize_kernel(psf)

            lambd *= 1.001

        print("%i/%i iterations completed" % ((it+1) * 5, iterations*5))

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

    cdef np.ndarray[DTYPE_t, ndim=3] u_masked = u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3], ...]
    cdef np.ndarray[DTYPE_t, ndim=3] im_masked = image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3], ...]

    cdef int i

    #cdef convolve FFT_valid = convolve(u[..., 0], psf[..., 0], "valid")
    #cdef convolve FFT_full = convolve(image[..., 0], psf[..., 0], "full")
    #cdef convolve FFT_kern_valid = convolve(u[..., 0], image[..., 0], "valid")

    # Set the nummber of threads after checking the memory
    cdef float free_RAM = psutil.virtual_memory()[4]
    cdef float footprint = (M + MK) * (N + NK) * 32 * 5 # resolution * bit depth * max concurrent copies
    cdef int threads = np.minimum(np.floor(free_RAM/footprint), CPU)


    with nogil, parallel(num_threads=threads):
        for i in prange(3):
            with gil:
                print("Channel %i - %i main threads" % (i, threads))
                _update_both_MM(u[..., i], image[..., i], psf[..., i], lambd, iterations, u_masked[..., i], im_masked[..., i], epsilon, blind, p)#, FFT_valid, FFT_full, FFT_kern_valid)

    u = u[pad:-pad, pad:-pad, ...]

    return [u.astype(np.float32), psf]

def richardson_lucy_MM(image, u, psf, lambd, iterations, epsilon, mask=None, blind=True, p=1):
    return  _richardson_lucy_MM(image, u, psf, lambd, iterations, epsilon, mask=None, blind=True, p=1)