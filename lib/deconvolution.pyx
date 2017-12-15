# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange
import matplotlib.pyplot as plt
import pyfftw

import multiprocessing


cdef int CPU = int(multiprocessing.cpu_count())

cdef extern from "math.h" nogil:
    float fabsf(float)
    float powf(float, float)
    int isnan(float)


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Activate debug checks for types and values - Set to False in production
cdef int DEBUG = False


cdef int check_nan_3D(float[:, :, :] array, int M, int N, int C) nogil:
    cdef size_t i, j, k
    cdef int out = 0

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(C):
                    if isnan(array[i, j, k]):
                        out += 1

    return out


cdef float norm_L2_3D(float[:, :] dim1, float[:, :]  dim2, int M, int N) nogil:
    cdef size_t i, j
    cdef float out = 0

    with nogil:
        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                    dim1[i, j] = powf((powf(dim1[i, j], 2) + powf(dim2[i, j], 2)), 0.5)

        for i in range(M):
            for j in range(N):
                out += powf(dim1[i, j], 2)

        out = powf(out, 0.5)

    return out


cdef float norm_L2_2D(float[:, :]  dim1, int M, int N) nogil:
    cdef size_t i, j
    cdef float out = 0

    with nogil:
        for i in range(M):
            for j in range(N):
                out += powf(dim1[i, j], 2)

        out = powf(out, 0.5)

    return out

cdef float best_param(float[:, :, :] image, float lambd, int M, int N):
    """
    Find the minimum acceptable epsilon to avoid a degenerate constant solution [2]
    Epsilon is the logarithmic norm prior  

    Reference :
    .. [1] https://pdfs.semanticscholar.org/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    .. [2] http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf

    :param image:
    :return:
    """

    cdef size_t i, j, k

    # Average the channels to compute only the parameters just once
    cdef np.ndarray[DTYPE_t, ndim=2] image_average = np.empty((M, N), dtype=DTYPE)

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                image_average[i, j] = 0
                for k in range(3):
                    image_average[i, j] += image[i, j, k] / 3

    # Compute the mean value
    cdef float image_mean = 0

    with nogil:
        for i in range(M):
            for j in range(N):
                image_mean += image_average[i, j]

        image_mean /= M*N

    # Compute omega
    cdef float omega = 2 * lambd * norm_L2_2D(image_average - image_mean, M, N) / (M*N)

    # Compute the gradient
    cdef float [:, :] gradx = np.empty((M, N), dtype=DTYPE)
    cdef float [:, :] grady = np.empty((M, N), dtype=DTYPE)
    cdef float [:, :] temp_buffer_2D = np.empty((M, N), dtype=DTYPE)

    grad2D(image_average, 0, M, N, gradx, temp_buffer_2D)
    grad2D(image_average, 1, M, N, grady, temp_buffer_2D)

    cdef float grad_mean = norm_L2_3D(gradx, grady, M, N) / (M*N)

    # Compute epsilon
    cdef float epsilon = (grad_mean / (np.exp(omega) - 1))**0.5 * 1.01

    #print(lambd, epsilon)
    return epsilon


cdef void _normalize_kernel(float[:, :, :] kern, int MK):
    """
    normalize a 3D kernel along its 2 first dimensions
    
    :param kern: 
    :param MK: 
    :return: 
    """

    cdef size_t i, j, k
    cdef float check = 0
    cdef float temp[3]
    temp[:] = [0., 0., 0.]

    with nogil:

        # Make negative values == 0
        for i in range(MK):
            for j in range(MK):
                for k in range(3):
                    if (kern[i, j, k] < 0):
                        kern[i, j, k] = - kern[i, j, k]

                    temp[k] += kern[i, j, k]

        # Make the sum of elements along the 2 first dimensions == 1
        for i in range(MK):
            for j in range(MK):
                for k in range(3):
                    kern[i, j, k] /= temp[k]

    if DEBUG:
        for i in range(MK):
            for j in range(MK):
                for k in range(3):
                    check += kern[i, j, k]

        if check - 3.0 > 1e-5 :
            raise ValueError("The kernel is not properly normalized, sum = %f" % check)



cpdef void normalize_kernel(np.ndarray[DTYPE_t, ndim=3] kern, int MK):
    # Expose the Cython function to Python
    _normalize_kernel(kern, MK)


cdef class convolve:
    cdef object fft_A_obj, fft_B_obj, ifft_obj,  output_array, A, B, C
    cdef int X, Y, M, N, offset_X, offset_Y, M_input, N_input, MK_input, NK_input

    def __cinit__(self, int M, int N, int MK, int NK, str domain):
        """
        Implements a convolution product using pyFFTW

        :param M:
        :param N:
        :param MK:
        :param NK:
        :param domain:
        :return:
        """

        # Takes an M×N image and an MK×MK kernel or second image
        self.M_input = M
        self.N_input = N
        self.MK_input = MK
        self.NK_input = NK

        # Compute the finals dimensions of the output
        if domain =="same":
            self.Y = self.M_input
            self.X = self.N_input
        elif domain == "valid":
            self.Y = self.M_input - self.MK_input + 1
            self.X = self.N_input - self.NK_input + 1
        elif domain == "full":
            self.Y = self.M_input + self.MK_input - 1
            self.X = self.N_input + self.NK_input - 1

        # Compute the dimensions of the FFT
        self.M = self.M_input + self.MK_input -1
        self.N = self.N_input + self.NK_input -1

        self.offset_Y = int(np.floor((self.M - self.Y)/2))
        self.offset_X = int(np.floor((self.N - self.X)/2))

        # Initialize the FFTW containers
        self.A = pyfftw.zeros_aligned((self.M, self.N), dtype=DTYPE, n=pyfftw.simd_alignment)
        self.B = pyfftw.zeros_aligned((self.M, self.N), dtype=DTYPE, n=pyfftw.simd_alignment)
        self.C = pyfftw.zeros_aligned((self.M, self.N), dtype=np.complex64, n=pyfftw.simd_alignment)

        self.fft_A_obj = pyfftw.builders.rfft2(self.A, s=(self.M, self.N), threads=CPU, auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.fft_B_obj = pyfftw.builders.rfft2(self.B, s=(self.M, self.N ), threads=CPU,  auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.ifft_obj = pyfftw.builders.irfft2(self.C, s=(self.M, self.N), threads=CPU, auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")


    def __call__(self, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B):

        # Fill the containers with input
        self.A[0:self.M_input, 0:self.N_input] = A
        self.B[0:self.MK_input, 0:self.NK_input] = B

        # Run the FFTs
        cdef float complex [:, :] Afft = self.fft_A_obj(self.A)
        cdef float complex [:, :] Bfft = self.fft_B_obj(self.B)
        cdef float complex [:, :] cfft = self.C

        # Multiply
        cdef size_t i, j
        with nogil, parallel(num_threads=CPU):
            for i in prange(self.M):
                for j in range(self.N):
                    cfft[i, j] = Afft[i, j] * Bfft[i, j]

        # Run the iFFT, slice and exit
        return self.ifft_obj(self.C)[self.offset_Y:self.offset_Y + self.Y, self.offset_X:self.offset_X + self.X]


cdef int sym_bound(int num_elem, int current_index) nogil:
    """
    Set the index accordingly to loop over an array dimension during a convolution to replace a symmetric padding
    
    Note : num_elem == last indice + 1
    
    :param padding: 
    :param num_elem: 
    :param current_index: 
    :return: 
    """

    if(current_index >= 0):
        if(current_index < num_elem):
            # main portion of the array
            return current_index

        else:
            # padding portion after
            return 2 * num_elem - current_index  -2
    else:
        # padding portion before
        return -current_index


cdef int period_bound(int num_elem, int current_index) nogil:
    """
    Set the index accordingly to loop over an array dimension during a convolution to replace a periodic padding
    
    Note : num_elem == last indice + 1
    
    :param padding: 
    :param num_elem: 
    :param current_index: 
    :return: 
    """

    if(current_index >= 0):
        if(current_index < num_elem):
            # main portion of the array
            return current_index

        else:
            # padding portion after
            return current_index - num_elem
    else:
        # padding portion before
        return num_elem + current_index


cdef void grad2D(float[:, :] u, int axis, int M, int N, float[:, :] out, float[:, :] temp_buffer_2D):
    """
    Convolve a 2D image with a separable kernel representing the 2nd order gradient on the neighbouring pixels with an 
    efficient approximation as described by [1]
    
    Reference
    --------
        [1] https://cdn.intechopen.com/pdfs-wm/39346.pdf
        [2] http://ieeexplore.ieee.org/document/8094858/#full-text-section
        [3] http://www.songho.ca/dsp/convolution/convolution.html
    
    :param u: 
    :param axis:
    :return: 
    """

    # Initialize the filter vectors
    # The whole 2D kernel can be reconstructed by the outer product in the same order

    # Set-up the default configuration to evaluate the gradient along the X direction
    cdef float vect_one[3]
    cdef float vect_two[3]

    cdef size_t i, j, nk = 0

    if axis == 0:
        # gradient along rows - y
        vect_one[:] = [-0.5, 0, 0.5]
        vect_two[:] = [-1/6.0, -2/3.0, -1/6.0]

    elif axis == 1:
        # gradient along columns - x
        vect_one[:] = [-1/6.0, -2/3.0, -1/6.0]
        vect_two[:] = [-0.5, 0, 0.5]

    with nogil:

        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                        temp_buffer_2D[i, j] = 0
                        for nk in range(3):
                            temp_buffer_2D[i, j] += vect_one[nk] * u[sym_bound(M, i-nk), j]

        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                        out[i, j] = 0
                        for nk in range(3):
                            out[i, j] += vect_two[nk] * temp_buffer_2D[i, sym_bound(N, j-nk+1)]

    #print("out z: Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(out), np.amax(out), np.mean(out), np.median(out), np.std(out)))
    #img = plt.imshow(out/np.amax(out))
    #plt.show()


cdef void grad3D(float[:, :, :] u, int axis, int M, int N, float[:, :, :] out, float[:, :, :] temp_buffer):
    """
    Convolve a 3D image with a separable kernel representing the 2nd order gradient on the 18 neighbouring pixels with an 
    efficient approximation as described by [1]
    
    Performs an spatial AND spectral gradient evaluation to remove all finds of noise, as described in [2]
    
    Reference
    --------
        [1] https://cdn.intechopen.com/pdfs-wm/39346.pdf
        [2] http://ieeexplore.ieee.org/document/8094858/#full-text-section
        [3] http://www.songho.ca/dsp/convolution/convolution.html
    
    :param u: 
    :param axis:
    :return: 
    """

    # Initialize the filter vectors
    # The whole 3D kernel can be reconstructed by the outer product in the same order

    # Set-up the default configuration to evaluate the gradient along the X direction
    cdef float vect_one[3]
    cdef float vect_two[3]
    cdef float vect_three[3]

    cdef int C = 3
    cdef size_t i, j, k, nk = 0

    if axis == 0:
        # gradient along rows - y
        vect_one[:] = [1/9.0, 0, -1/9.0]
        vect_two[:] = [1/2.0, 2, 1/2.0]
        vect_three[:] = [1/4.0, 1, 1/4.0]

    elif axis == 1:
        # gradient along columns - x
        vect_one[:] = [1/2.0, 2, 1/2.0]
        vect_two[:] = [1/9.0, 0, -1/9.0]
        vect_three[:] = [1/4.0, 1, 1/4.0]

    elif axis == 2:
        # gradient along depth - z
        vect_one[:] = [1/4.0, 1, 1/4.0]
        vect_two[:] = [1/2.0, 2, 1/2.0]
        vect_three[:] = [1/9.0, 0, -1/9.0]

    with nogil:
        # Cleanup the buffers
        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                    for k in range(C):
                        out[i, j, k] = 0
                        temp_buffer[i, j, k] = 0

        # Rows gradient
        with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for j in range(N):
                    for k in range(C):
                        for nk in range(3):
                            out[i, j, k] += vect_one[nk] * u[i-nk+1, j, k]

        # Columns gradient
        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(1, N-1):
                    for k in range(C):
                        for nk in range(3):
                            temp_buffer[i, j, k] += vect_two[nk] * out[i, j-nk+1, k]

        # Depth gradients
        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                    for k in range(C):
                        # ensure out is clean
                        out[i, j, k] = 0
                        for nk in range(3):
                            out[i, j, k] += vect_three[nk] * temp_buffer[i, j, period_bound(C, k-nk+1)]

    # Debug
    #print("out z: Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(out), np.amax(out), np.mean(out), np.median(out), np.std(out)))
    #img = plt.imshow(out/np.amax(out))
    #plt.show()

    if DEBUG:
        print("grad3D passed")

        if check_nan_3D(out, M, N, 3) != 0:
            raise ValueError("Grad3D contains NaN")

        if not np.any(out):
            print(M, N)
            raise ValueError("Grad3D is all null")


from scipy.signal import fftconvolve

cdef np.ndarray[DTYPE_t, ndim=2] _convolve_image(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf, convolve FFT_valid, convolve FFT_full):
    cdef np.ndarray[DTYPE_t, ndim=2] error
    error = FFT_valid(u, psf)
    #error = fftconvolve(u, psf, "valid").astype(DTYPE)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                error[i, j] -= image[i, j]

    return FFT_full(error, np.rot90(psf, 2))
    #return fftconvolve(error, np.rot90(psf, 2), "full")


cdef np.ndarray[DTYPE_t, ndim=2] _convolve_kernel(np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[DTYPE_t, ndim=2] psf, convolve FFT_valid, convolve FFT_kern_valid):
    cdef np.ndarray[DTYPE_t, ndim=2] error

    error = FFT_valid(u, psf)
    #error = fftconvolve(u, psf, "valid").astype(DTYPE)

    cdef int M = image.shape[0]
    cdef int N = image.shape[1]
    cdef int i, j

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                error[i, j] -= image[i, j]


    return FFT_kern_valid(np.rot90(u, 2), error)
    #return fftconvolve(np.rot90(u, 2), error, "valid").astype(DTYPE)


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


cdef np.ndarray[DTYPE_t, ndim=3] gradTVEM(np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] ut, float epsilon, float tau):
    # http://ieeexplore.ieee.org/document/8094858/#full-text-section
    # https://cdn.intechopen.com/pdfs-wm/39346.pdf
    # https://www.intechopen.com/books/matlab-a-fundamental-tool-for-scientific-computing-and-engineering-applications-volume-3/convolution-kernel-for-fast-cpu-gpu-computation-of-2d-3d-isotropic-gradients-on-a-square-cubic-latti

    cdef int M = u.shape[0]
    cdef int N = u.shape[1]
    cdef int C = 2
    cdef float[:, :, :] gradx, grady, gradz
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.empty((M, N, 3), dtype=DTYPE)


    cdef size_t i, j, k

    gradx = np.empty((M, N, 3), dtype=DTYPE)
    grady = np.empty((M, N, 3), dtype=DTYPE)
    gradz = np.empty((M, N, 3), dtype=DTYPE)

    cdef float [:, :, :] temp_buffer = np.empty_like(u)

    grad3D(u, 1, M, N, gradx, temp_buffer)
    grad3D(u, 0, M, N, grady, temp_buffer)
    grad3D(u, 2, M, N, gradz, temp_buffer)

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(3):
                    #print(abs(gradx[i, j, k]))
                    out[i, j, k] = fabsf(gradx[i, j, k]) + fabsf(grady[i, j, k]) + fabsf(gradz[i, j, k]) + epsilon

    # Debug
    # grad = np.gradient(u, edge_order=2)
    # out = np.abs(grad[0]) + np.abs(grad[1]) + epsilon
    # print("Gradx u : Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(gradx), np.amax(gradx), np.mean(gradx), np.median(gradx), np.std(gradx)))
    # print("Grady u : Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(grady), np.amax(grady), np.mean(grady), np.median(grady), np.std(grady)))
    # print("GradTV u : Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(out), np.amax(out), np.mean(out), np.median(out), np.std(out)))
    # img = plt.imshow(out/np.amax(out))
    # plt.show()

    if DEBUG:
        if check_nan_3D(out, M, N, 3) != 0:
            raise ValueError("TV u contains NaN")

        if not np.any(out):
            raise ValueError("TV u is all null")

    grad3D(ut, 1, M, N, gradx, temp_buffer)
    grad3D(ut, 0, M, N, grady, temp_buffer)
    grad3D(ut, 2, M, N, gradz, temp_buffer)

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(3):
                    out[i, j, k] /= fabsf(gradx[i, j, k]) + fabsf(grady[i, j, k]) + fabsf(gradz[i, j, k]) + epsilon + tau

    # Debug
    # grad = np.gradient(ut, edge_order=2)
    # out /= np.abs(grad[0]) + np.abs(grad[1]) + epsilon + 1e-1
    # print("Gradx ut : Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(gradx), np.amax(gradx), np.mean(gradx), np.median(gradx), np.std(gradx)))
    # print("Grady ut : Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(grady), np.amax(grady), np.mean(grady), np.median(grady), np.std(grady)))
    # print("GradTV ut : Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(out), np.amax(out), np.mean(out), np.median(out), np.std(out)))
    # img = plt.imshow(out/np.amax(out))
    # plt.show()

    if DEBUG:
        print("gradTVEM passed")

        if check_nan_3D(out, M, N, 3) != 0:
            raise ValueError("TV ut contains NaN")

        if not np.any(out):
            raise ValueError("TV ut is all null")

    return out


"""
cdef float[:] coeff_RGB(float[:, :, :] u, float[:, :, :] gradu, int M, int N) nogil:

    cdef size_t i, j, k
    cdef float out = 0

    dt[chan] = u_step[step] * (amax_2D(u[..., chan], u_M, u_N) + 1 / (M*N) ) / (amax_2D(np.abs(gradu[..., chan]), u_M, u_N) + 1e-31)

    with nogil, parallel(num_threads=CPU):
        for i in range(M):
            for j in range(N):
                if array[i, j] > out:
                    out = array[i, j]

    return out
"""


cdef void _richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                              float lambd, float tau, float step_factor, int M, int N, int C, int MK, int blind=True):
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


    cdef float eps, alpha, max_gradu, abs_gradu

    cdef int u_M = u.shape[0]
    cdef int u_N = u.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] gradk
    cdef np.ndarray[DTYPE_t, ndim=3] ut
    cdef np.ndarray[DTYPE_t, ndim=3] gradTV, gradTVz
    cdef np.ndarray[DTYPE_t, ndim=3] im_convo = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] gradu = np.empty_like(u, dtype=DTYPE)
    cdef float dt[3]

    gradk = np.zeros_like(psf[..., 0])

    cdef convolve FFT_valid = convolve(u_M, u_N, MK, MK, "valid")
    cdef convolve FFT_full = convolve(M, N, MK, MK, "full")
    cdef convolve FFT_kern_valid = convolve(u_M, u_N, M, N, "valid")

    cdef float k_step[3]
    cdef float u_step[3]
    cdef size_t step, it, itt, i, j, chan
    cdef int iterations[3]
    cdef int main_iter

    iterations[:] = [MK, MK*2, MK*4]
    u_step[:] = [step_factor, step_factor/2, step_factor/4]
    k_step[:] = [step_factor, step_factor/2, step_factor/4]

    if blind:
        main_iter = 3
    else:
        main_iter = 2

    for step in range(main_iter):
        print("Step %i/%i : %i iterations to perform" % (step+1, main_iter, iterations[step]*5))
        for it in range(iterations[step]):
            ut = u.copy()
            lambd = min([lambd, 50000])

            for itt in range(5):
                # Image update
                lambd = min([lambd, 50000])

                eps = best_param(u, lambd, u_M, u_N)
                gradTV = gradTVEM(u, ut, eps, tau)

                for chan in range(3):
                    im_convo[..., chan] = _convolve_image(u[..., chan], image[..., chan], psf[..., chan], FFT_valid, FFT_full)

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for j in range(u_N):
                            for chan in range(C):
                                gradu[i, j, chan] = lambd *  im_convo[i, j, chan] + gradTV[i, j, chan]

                for chan in range(3):
                    dt[chan] = u_step[step] * (np.amax(u[..., chan]) + 1 / (M*N) ) / (np.amax(np.abs(gradu[..., chan])) + 1e-31)

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for j in range(u_N):
                            for chan in range(C):
                                u[i, j, chan] -= dt[chan] * gradu[i, j, chan]

                if DEBUG:
                    if check_nan_3D(u, M, N, 3) != 0:
                        raise ValueError("u is NaN")

                    if not np.any(u):
                        raise ValueError("u is all null")

                if blind:
                    # PSF update
                    for chan in range(C):
                        gradk = _convolve_kernel(u[..., chan], image[..., chan], psf[..., chan], FFT_valid, FFT_kern_valid)
                        alpha = k_step[step] * (np.amax(psf[..., chan]) + 1 / (MK*MK)) / np.amax(np.abs(gradk[..., chan]) + 1e-31)
                        psf[..., chan] -= alpha * gradk

                    _normalize_kernel(psf, MK)

                if DEBUG:
                    if check_nan_3D(psf, MK, MK, 3) != 0:
                        raise ValueError("psf is NaN")

                    if not np.any(psf):
                        raise ValueError("PSF is all null")

            lambd *= 1.001


        print("%i/%i iterations completed\n" % ((step+1)*(it+1)*(itt+1), 2*6*5*MK))



        #img = plt.imshow(gradTV[..., 0]/np.amax(gradTV[..., 0]))
        #plt.show()


def richardson_lucy_MM(image, u, psf, lambd, tau, step, M, N, C, MK, blind=True):
    # Expose the Cython function to Python
    _richardson_lucy_MM(image, u, psf, lambd, tau, step, M, N, C, MK, blind=blind)
