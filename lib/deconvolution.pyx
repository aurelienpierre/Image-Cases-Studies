# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
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

cdef int CPU = int(multiprocessing.cpu_count())

cdef extern from "stdlib.h":
    float abs(float x) nogil


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

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
    cdef int X, Y, M, N, offset_X, offset_Y

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

        self.offset_Y = int(np.floor((self.M - self.Y)/2))
        self.offset_X = int(np.floor((self.N - self.X)/2))

        A = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
        B = np.pad(B, ((0, M - 1), (0, N - 1)), mode='constant')

        self.fft_A_obj = pyfftw.builders.rfft2(A, s=(self.M, self.N), threads=CPU, auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.fft_B_obj = pyfftw.builders.rfft2(B, s=(self.M, self.N ), threads=CPU,  auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.ifft_obj = pyfftw.builders.irfft2(self.fft_B_obj.output_array , s=(self.M, self.N), threads=CPU,
                                               auto_align_input=True, auto_contiguous=True, planner_effort="FFTW_ESTIMATE")

        self.output_array = self.ifft_obj.output_array

    def __call__(self, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B):
        """
        The lock mechanism makes it thread-safe
        :param A:
        :param B:
        :return:
        """

        MK =  B.shape[0]
        NK = B.shape[1]
        M = A.shape[0]
        N = A.shape[1]

        A = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
        B = np.pad(B, ((0, M - 1), (0, N - 1)), mode='constant')

        return self.ifft_obj(self.fft_A_obj(A) * self.fft_B_obj(B))[self.offset_Y:self.offset_Y + self.Y, self.offset_X:self.offset_X + self.X]


cdef class convolve3D:
    cdef object fft_A_obj, fft_B_obj, ifft_obj,  output_array
    cdef int X, Y, M, N, offset_X, offset_Y

    def __cinit__(self, np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=3] B, str domain):
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

        self.offset_Y = int(np.floor((self.M - self.Y)/2))
        self.offset_X = int(np.floor((self.N - self.X)/2))


        A = pad_image(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
        A = np.dstack([A, np.zeros_like(A[..., 1]), np.zeros_like(A[..., 1])])
        B = pad_image(B, ((0, M - 1), (0, N - 1)), mode='constant')
        B = np.dstack([B, np.zeros_like(B[..., 1]), np.zeros_like(B[..., 1])])

        self.fft_A_obj = pyfftw.builders.rfftn(A, s=(self.M, self.N, 5), threads=CPU, auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.fft_B_obj = pyfftw.builders.rfftn(B, s=(self.M, self.N, 5), threads=CPU,  auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.ifft_obj = pyfftw.builders.irfftn(self.fft_B_obj.output_array , s=(self.M, self.N, 5), threads=CPU,
                                               auto_align_input=True, auto_contiguous=True, planner_effort="FFTW_ESTIMATE")

        self.output_array = self.ifft_obj.output_array

    def __call__(self, np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=3] B):
        """
        The lock mechanism makes it thread-safe
        :param A:
        :param B:
        :return:
        """

        MK =  B.shape[0]
        NK = B.shape[1]
        M = A.shape[0]
        N = A.shape[1]

        A = pad_image(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
        A = np.dstack([A, np.zeros_like(A[..., 1]), np.zeros_like(A[..., 1])])
        B = pad_image(B, ((0, M - 1), (0, N - 1)), mode='constant')
        B = np.dstack([B, np.zeros_like(B[..., 1]), np.zeros_like(B[..., 1])])

        return self.ifft_obj(self.fft_A_obj(A) * self.fft_B_obj(B))[self.offset_Y:self.offset_Y + self.Y, self.offset_X:self.offset_X + self.X, 1:4]


cdef float [:, :, :]  conv3(np.ndarray[DTYPE_t, ndim=3] u, int axis_one, int axis_two, int axis_three):
    """
    Convolve a 3D image with a separable kernel representing the 2nd order gradient on the 18 neighbours
    :param u: 
    :param axis_one: 
    :param axis_two: 
    :param axis_three: 
    :return: 
    """

    cdef float[:] first_dim = np.array([1/9, 0, -1/9], dtype=DTYPE)
    cdef float[:] second_dim = np.array([1/2, 2, 1/2], dtype=DTYPE)
    cdef float [:] third_dim = np.array([1/4, 1, 1/4], dtype=DTYPE)
    cdef float [:] vect_one = np.array([1/9, 0, -1/9], dtype=DTYPE)
    cdef float [:] vect_two = np.array([1/2, 2, 1/2], dtype=DTYPE)
    cdef float [:] vect_three = np.array([1/4, 1, 1/4], dtype=DTYPE)


    cdef np.ndarray[DTYPE_t, ndim=3] u_pad = pad_image(u, (2, 2), "wrap")
    cdef np.ndarray[DTYPE_t, ndim=3] u_conv = np.dstack((u_pad, u_pad[..., 1], u_pad[..., 0]))
    del u_pad
    cdef float [:, :, :] out = np.zeros_like(u_conv)


    if axis_one == 1:
        vect_one = second_dim
    elif axis_one == 2:
        vect_one = third_dim

    if axis_two == 0:
        vect_two = first_dim
    elif axis_three == 2:
        vect_two = third_dim

    if axis_three == 0:
        vect_three = first_dim
    elif axis_three == 1:
        vect_three = second_dim


    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int C = u.shape[2]

    cdef int i, j, k

    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                for k in range(C):
                    out[i+2, j, k] += vect_one[0] * u_conv[i, j, k]
                    out[i+2, j, k] += vect_one[1] * u_conv[i+1, j, k]
                    out[i+2, j, k] += vect_one[2] * u_conv[i+2, j, k]
                    u_conv[i, j, k] = 0

        for i in prange(M, schedule="guided"):
            for j in range(N):
                for k in range(C):
                    u_conv[i, j+2, k] += vect_two[0] * out[i, j, k]
                    u_conv[i, j+2, k] += vect_two[1] * out[i, j+1, k]
                    u_conv[i, j+2, k] += vect_two[2] * out[i, j+2, k]
                    out[i, j, k] = 0

        for i in prange(M, schedule="guided"):
            for j in range(N):
                for k in range(C):
                    out[i, j, k+2] += vect_three[0] * u_conv[i, j, k]
                    out[i, j, k+2] += vect_three[1] * u_conv[i, j, k+1]
                    out[i, j, k+2] += vect_three[2] * u_conv[i, j, k+2]

    return out[2:2+M, 2:2+N, 0:3]



cdef np.ndarray[DTYPE_t, ndim=2] _convolve_image(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf, convolve FFT_valid, convolve FFT_full):
    cdef np.ndarray[DTYPE_t, ndim=2] error
    error = FFT_valid(u, psf)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                error[i, j] -= image[i, j]

    return FFT_full(error, np.rot90(psf, 2))


cdef np.ndarray[DTYPE_t, ndim=2] _convolve_kernel(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf, convolve FFT_valid, convolve FFT_kern_valid):
    cdef np.ndarray[DTYPE_t, ndim=2] error

    error = FFT_valid(u, psf)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                error[i, j] -= image[i, j]

    return FFT_kern_valid(np.rot90(u, 2), error)


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


cdef np.ndarray[DTYPE_t, ndim=3] gradTVEM(np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] ut, np.ndarray[DTYPE_t, ndim=3] gradx, np.ndarray[DTYPE_t, ndim=3] grady, float eps, convolve3D FFT_3D):
    # https://cdn.intechopen.com/pdfs-wm/39346.pdf
    # https://www.intechopen.com/books/matlab-a-fundamental-tool-for-scientific-computing-and-engineering-applications-volume-3/convolution-kernel-for-fast-cpu-gpu-computation-of-2d-3d-isotropic-gradients-on-a-square-cubic-latti

    cdef float [:, :, :] gradTVx, gradTVy
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros_like(u)

    gradTVx = conv3(u, 0, 1, 2)
    gradTVy = conv3(u, 1, 0, 2)

    cdef int i, j, k, M, N, C
    M = u.shape[0]
    N = u.shape[1]
    C = u.shape[2]

    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                for k in range(C):
                    out[i, j, k] = (abs(gradTVx[i, j, k]) + abs(gradTVy[i, j, k]) + eps)


    gradTVx = conv3(ut, 0, 1, 2)
    gradTVy = conv3(ut, 1, 0, 2)


    with nogil, parallel(num_threads=CPU):
        for i in prange(M, schedule="guided"):
            for j in range(N):
                for k in range(C):
                    out[i, j, k] /= abs(gradTVx[i, j, k]) + abs(gradTVy[i, j, k]) + eps

    return out



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
    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int pad = np.floor(MK / 2).astype(int)

    u = pad_image(u, (pad, pad)).astype(np.float32)

    if mask == None:
        mask_i = [0, M + 2 * pad, 0, N + 2 * pad]
        mask_u = [0, M + 2 * pad, 0, N + 2 * pad]
    else:
        mask_u = [mask[0] - pad, mask[1] + pad, mask[2] - pad, mask[3] + pad]
        mask_i = mask

    cdef float k_step = epsilon
    cdef float u_step = epsilon
    cdef float eps, alpha, max_gradu, abs_gradu
    cdef float[:] dt = np.array([0, 0, 0], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2] gradk
    cdef np.ndarray[DTYPE_t, ndim=3] gradV, ut, gradV_bot
    cdef np.ndarray[DTYPE_t, ndim=3] im_convo = np.zeros_like(u)
    cdef float[:, :, :] gradu = np.zeros_like(u)

    gradk = np.zeros_like(psf[..., 0])

    cdef int i, j, chan

    # Compute the 3D gradients with an efficient method. From : https://cdn.intechopen.com/pdfs-wm/39346.pdf
    cdef float w1 = 2/9
    cdef float w2 = 1/18
    cdef float w3 = 1/72

    cdef float [:, :] kernel1, kernel2

    kernel1 = np.array([
        [-w3, 0, w3],
        [-w2, 0, w2],
        [-w3, 0, w3]]).astype(DTYPE)

    kernel2 = np.array([
        [-w2, 0, w2],
        [-w1, 0, w1],
        [-w2, 0, w2]]).astype(DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=3] kern = np.dstack((kernel1, kernel2, kernel1))
    cdef np.ndarray[DTYPE_t, ndim=3] gradx = - np.transpose(kern, axes=(1, 0, 2))
    cdef np.ndarray[DTYPE_t, ndim=3] grady = np.transpose(gradx, axes=(1, 0, 2))
    cdef np.ndarray[DTYPE_t, ndim=3] gradz = np.transpose(gradx, axes=(2, 1, 0))

    # [1/9, 0, -1/9], [1/2, 2, 1/2], [1/4, 1, 1/4]

    print("System profiling…")
    cdef convolve FFT_valid = convolve(u[..., 0], psf[..., 0], "valid")
    cdef convolve FFT_full = convolve(image[..., 0], psf[..., 0], "full")
    cdef convolve FFT_masked_valid = convolve(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3], 0], psf[..., 0], "valid")
    cdef convolve FFT_kern_valid = convolve(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3], 0], image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3], 0], "valid")
    cdef convolve3D FFT_3D = convolve3D(u, gradx, "same")
    print("Profiling done !")


    for it in range(iterations):
        ut = u.copy()
        lambd = min([lambd, 50000])
        eps = best_param(u.mean(axis=2), lambd, p=1)

        for itt in range(5):
            # Image update
            lambd = min([lambd, 50000])
            gradV = gradTVEM(u, ut, gradx, grady, eps, FFT_3D)

            for chan in range(3):
                im_convo[..., chan] = _convolve_image(u[..., chan], image[..., chan], psf[..., chan], FFT_valid, FFT_full)

            with nogil, parallel(num_threads=CPU):
                for i in prange(M, schedule="guided"):
                    for j in range(N):
                        for chan in range(3):
                            gradu[i, j, chan] = lambd *  im_convo[i, j, chan] + gradV[i, j, chan]


            with nogil, parallel(num_threads=CPU):
                for chan in prange(3):
                    with gil:
                        dt[chan] = u_step * (np.amax(u[..., chan]) + 1 / (M*N)) / (np.amax(np.abs(gradu)) + 1e-31)


            with nogil, parallel(num_threads=CPU):
                for i in prange(M, schedule="guided"):
                    for j in range(N):
                        for chan in range(3):
                            u[i, j, chan] -= dt[chan] * gradu[i, j, chan]

                            if u[i, j, chan] > 1:
                                u[i, j, chan] = 1

                            elif u[i, j, chan] < 0:
                                u[i, j, chan] = 0


            if blind:
                # PSF update
                for chan in range(3):
                    gradk = _convolve_kernel(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3], chan],
                                             image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3], chan],
                                             psf[..., chan], FFT_masked_valid, FFT_kern_valid)
                    alpha = k_step * (np.amax(psf[..., chan]) + 1 / (MK*NK)) / np.amax(np.abs(gradk) + 1e-31)
                    psf[..., chan] -= alpha * gradk
                    psf[..., chan] = _normalize_kernel(psf[..., chan])

            lambd *= 1.001

        print("%i/%i iterations completed" % ((it+1) * 5, iterations*5))


    u = u[pad:-pad, pad:-pad, ...]

    return [u.astype(np.float32), psf]

def richardson_lucy_MM(image, u, psf, lambd, iterations, epsilon, mask=None, blind=True, p=1):
    return  _richardson_lucy_MM(image, u, psf, lambd, iterations, epsilon, mask=mask, blind=blind, p=p)
