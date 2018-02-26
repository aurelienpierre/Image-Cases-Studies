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
import pyfftw
import multiprocessing

cdef int CPU = int(multiprocessing.cpu_count())

cdef extern from "math.h" nogil:
    float fabsf(float)
    float powf(float, float)
    float expf(float)
    int isnan(float)
    float atan2f(float, float)
    float cosf(float)
    float sinf(float)


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef inline float norm_L2_2D(float[:, :] array, int M, int N) nogil:
    """L2 vector norm of a 2D array. Outputs a scalar."""

    cdef int i, j
    cdef float out = 0

    for i in range(M):
        for j in range(N):
            out += powf(array[i, j], 2)

    out = powf(out, 0.5)

    return out


cdef inline float norm_L2_3D(float [:, :, :] array, int M, int N) nogil:
    """L2 vector norm of a 3D array. The L2 norm is computed successively on the 3rd axis, then on the resulting 2D array."""

    cdef int i, j, k
    cdef float out = 0

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(3):
                    out += powf(array[i, j, k], 2)

    out = powf(out, 0.5)

    return out


cdef inline void amax(float[:, :, :] array, int M, int N, float out[3]) nogil:
    """Compute the max of every channel and output a 3D vector"""

    cdef int i, j, k
    cdef float temp
    out[0] = array[0, 0, 0]
    out[1] = array[0, 0, 1]
    out[2] = array[0, 0, 2]

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    temp = array[i, j, k]
                    if temp > out[k]:
                        out[k] = temp


cdef inline void amax_abs(float[:, :, :] array, int M, int N, float out[3]) nogil:
    """Compute the max of the absolute value of every channel and output a 3D vector"""

    cdef int i, j, k
    cdef float temp
    out[0] = fabsf(array[0, 0, 0])
    out[1] = fabsf(array[0, 0, 1])
    out[2] = fabsf(array[0, 0, 2])

    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    temp = fabsf(array[i, j, k])
                    if temp > out[k]:
                        out[k] = temp


cdef inline float amax_abs_2D(float[:, :] array, int M, int N) nogil:
    """Compute the max of the absolute value of every channel and output a float"""

    cdef int i, j
    cdef float temp
    cdef float out = fabsf(array[0, 0])

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                temp = fabsf(array[i, j])
                if temp > out:
                    out = temp

    return out
    
    
def best_param(float[:, :, :] image, float lambd, int M, int N):
    return _best_param(image, lambd, M, N)
    

cdef inline float _best_param(float[:, :, :] image, float lambd, int M, int N):
    """
    Find the minimum acceptable epsilon to avoid a degenerate constant solution [2]
    Epsilon is the logarithmic norm prior  

    Reference :
    .. [1] https://pdfs.semanticscholar.org/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    .. [2] http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf
    """

    cdef int i, j, k

    # Compute the mean value
    cdef float image_mean = 0

    with nogil:
        for i in range(M):
            for j in range(N):
                for k in range(3):
                    image_mean += image[i, j, k]

        image_mean /= M*N*3
        
        
    # Compute the gradient
    cdef float grad_mean = grad2D(image, M, N) / (M*N*3)
                
    cdef float norm_diff = 0

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(3):
                    norm_diff += powf(image[i, j, k] - image_mean, 2)

    norm_diff = powf(norm_diff, 0.5)

    # Compute omega
    cdef float omega = 2 * lambd * norm_diff / (M*N*3)
    
    # Compute epsilon
    cdef float epsilon = powf((grad_mean) / (expf(omega) - 1), 0.5)
    
    print("%1.12f, %f" % (epsilon, lambd))

    """
    image_average = np.mean(image, axis=2)
    grad_mean = np.sum(np.linalg.norm(np.gradient(image_average)))/ (M*N)
    omega = 2 * lambd * np.linalg.norm(image_average - image_average.mean()) / (M*N)
    epsilon = np.sqrt((grad_mean) / (np.exp(omega) - 1))* 1.001
    
    print("%1.12f, %f" % (epsilon, lambd))
    """
    
    return epsilon


cdef inline void _normalize_kernel(float[:, :, :] kern, int MK):
    """Normalizes a 3D kernel along its 2 first dimensions"""

    cdef int i, j, k
    cdef float check = 0
    cdef float temp[3]
    temp[:] = [0., 0., 0.]

    with nogil:
        # Make negative values == 0
        for k in range(3):
            for i in range(MK):
                for j in range(MK):
                    if (kern[i, j, k] < 0):
                        kern[i, j, k] = - kern[i, j, k]

                    temp[k] += kern[i, j, k]

        # Make the sum of elements along the 2 first dimensions == 1
        for k in range(3):
            for i in range(MK):
                for j in range(MK):
                    kern[i, j, k] /= temp[k]


cpdef void normalize_kernel(np.ndarray[DTYPE_t, ndim=3] kern, int MK):
    # Expose the Cython function to Python
    _normalize_kernel(kern, MK)


cdef class convolve:
    cdef object fft_A_obj, fft_B_obj, ifft_obj,  output_array, A, B, C
    cdef int X, Y, M, N, offset_X, offset_Y, M_input, N_input, MK_input, NK_input, Mfft, Nfft

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
        if domain == "valid":
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


        self.fft_A_obj = pyfftw.builders.rfft2(self.A, s=(self.M, self.N), threads=CPU, auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")
        self.fft_B_obj = pyfftw.builders.rfft2(self.B, s=(self.M, self.N ), threads=CPU,  auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")

        self.C = pyfftw.zeros_aligned(self.fft_A_obj.output_shape, dtype=np.complex64, n=pyfftw.simd_alignment)
        self.ifft_obj = pyfftw.builders.irfft2(self.fft_A_obj.output_array, s=(self.M, self.N), threads=CPU, auto_align_input=True,
                                               auto_contiguous=True, planner_effort="FFTW_ESTIMATE")

        self.Mfft = self.fft_A_obj.output_shape[0]
        self.Nfft = self.fft_A_obj.output_shape[1]


    def __call__(self, float[:, :] A, float[:, :] B):

        # Fill the containers with input
        self.A[0:self.M_input, 0:self.N_input] = A
        self.B[0:self.MK_input, 0:self.NK_input] = B

        # Run the FFTs
        cdef float complex [:, :] Afft = self.fft_A_obj(self.A)
        cdef float complex [:, :] Bfft = self.fft_B_obj(self.B)
        cdef float complex [:, :] cfft = self.C

        # Multiply
        cdef int i, j

        with nogil, parallel(num_threads=CPU):
            for i in prange(self.Mfft):
                for j in range(self.Nfft):
                    cfft[i, j] = Afft[i, j] * Bfft[i, j]

        # Run the iFFT, slice and exit
        cdef np.ndarray[DTYPE_t, ndim=2] out = self.ifft_obj(self.C)

        return out[self.offset_Y:self.offset_Y + self.Y, self.offset_X:self.offset_X + self.X]


cdef inline float grad2D(float[:, :, :] u, int M, int N) nogil:
    """Compute the L2,2 norm of the gradient using an average of the forward and backward finite difference"""

    cdef:
        int i, j, k
        float udx, udy, udxdy, udydx, dxdy, uxy, out
        float temp[3]
    
    out = 0
        
    dxdy = 1 / powf(2, 0.5) # Distance over the diagonal

    with parallel(num_threads=CPU):
        for i in prange(1, M-1):
            for j in range(1, N-1):
                for k in range(3):
                    uxy = u[i, j, k] * 2
                    udx = uxy - u[i-1, j, k] - u[i+1, j, k] # Warning : this is 2 times the differential
                    udy = uxy - u[i, j-1, k] - u[i, j-1, k]
                    
                    udxdy = (uxy - u[i-1, j-1, k] - u[i+1, j+1, k]) / dxdy
                    udydx = (uxy - u[i-1, j+1, k] - u[i+1, j-1, k]) / dxdy
                                        
                    out += powf(udx, 2) + powf(udy, 2)+ powf(udxdy, 2) + powf(udydx, 2)
                
    return powf(out, 0.5)


cdef inline void gradTVEM(float[:, :, :] u, float[:, :, :] ut, float epsilon, float tau, float[:, :, :] im_convo, float lambd, float[:, :, :] out, float[3] max_u, int M, int N, int neighbours) nogil:
    """Compute the L1 norm of the gradient using an average of the forward and backward finite difference over the main axis and the diagonals"""
    
    cdef:
        int i, j, k
        float udx, udy, udxdy, udydx, utdx, utdy, utdxdy, utdydx, uxy, utxy, dxdy, temp
        
    epsilon = 2 * epsilon
    tau = 2 * tau
    max_u[0] = 0
    max_u[1] = 0
    max_u[2] = 0
    
    
    ## In this section we treat only the inside of the picture. See the next section for edges and boundaries exceptions
        
        
    if neighbours == 8:
        # Evaluate the total variation on 8 neighbours : direct directions and diagonals, with a bilateral approximation
        
        dxdy = 1 / powf(2, 0.5) # Distance over the diagonal

        with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for j in range(1, N-1):
                    for k in range(3):
                        uxy = u[i, j, k] * 2
                        udx = uxy - u[i-1, j, k] - u[i+1, j, k] # Warning : this is 2 times the differential
                        udy = uxy - u[i, j-1, k] - u[i, j-1, k]
                        
                        udxdy = (uxy - u[i-1, j-1, k] - u[i+1, j+1, k]) / dxdy
                        udydx = (uxy - u[i-1, j+1, k] - u[i+1, j-1, k]) / dxdy
                        
                        utxy = ut[i, j, k] * 2
                        utdx = (utxy - ut[i-1, j, k] - ut[i+1, j, k]) / dxdy
                        utdy = (utxy - ut[i, j-1, k] - ut[i, j-1, k]) / dxdy
                        
                        utdxdy = utxy - ut[i-1, j-1, k] - ut[i+1, j+1, k]
                        utdydx = utxy - ut[i-1, j+1, k] - ut[i+1, j-1, k]
                                            
                        temp  = (udx + udy + udxdy + udydx) / (fabsf(udx) + fabsf(udy)+ fabsf(udxdy) + fabsf(udydx) + epsilon) / (fabsf(utdx) + fabsf(utdy)+ fabsf(utdxdy) + fabsf(utdydx) + epsilon + tau) / 4 / lambd +  im_convo[i, j, k]
                        
                        out[i, j, k] = temp
                        temp = fabsf(temp)
                        
                        if temp > max_u[k]:
                            max_u[k] = temp
                            
                        
    elif neighbours == 4:
        # Evaluate the total variation on 4 neighbours : direct directions, with a bilateral approximation
        
        with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for j in range(1, N-1):
                    for k in range(3):
                        uxy = u[i, j, k] * 2
                        udx = uxy - u[i-1, j, k] - u[i+1, j, k] # Warning : this is 2 times the differential
                        udy = uxy - u[i, j-1, k] - u[i, j-1, k]
                        
                        utxy = ut[i, j, k] * 2
                        utdx = utxy - ut[i-1, j, k] - ut[i+1, j, k]
                        utdy = utxy - ut[i, j-1, k] - ut[i, j-1, k]
    
                        temp = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 4 / lambd +  im_convo[i, j, k]
                        
                        out[i, j, k] = temp
                        temp = fabsf(temp)
                        
                        if temp > max_u[k]:
                            max_u[k] = temp
                        
                        
    elif neighbours == 2:
        # Evaluate the total variation on 2 neighbours : simple backward difference
        
        with parallel(num_threads=CPU):
            for i in prange(1, M):
                for j in range(1, N):
                    for k in range(3):
                        uxy = u[i, j, k]
                        udx = uxy - u[i-1, j, k]
                        udy = uxy - u[i, j-1, k]
                        
                        utxy = ut[i, j, k]
                        utdx = utxy - ut[i-1, j, k]
                        utdy = utxy - ut[i, j-1, k]
        
                        temp = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd +  im_convo[i, j, k]
                        
                        out[i, j, k] = temp
                        temp = fabsf(temp)
                        
                        if temp > max_u[k]:
                            max_u[k] = temp
                        
                        
    ## In the following section, we treat independently the borders and the edges with custom differences in the relevant directions
    
    # First row
    with parallel(num_threads=CPU):
            for j in prange(1, N-1):
                for k in range(3):
                    uxy = u[0, j, k]
                    udx = uxy - u[1, j, k]
                    udy = uxy - (u[0, j-1, k] + u[0, j+1, k])/2
                    
                    utxy = ut[0, j, k]
                    utdx = utxy - ut[1, j, k]
                    utdy = utxy - (ut[0, j-1, k] + ut[0, j+1, k])/2
                    
                    out[0, j, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd +  im_convo[0, j, k]
                    
    # Last row
    with parallel(num_threads=CPU):
            for j in prange(1, N-1):
                for k in range(3):
                    uxy = u[M-1, j, k]
                    udx = uxy - u[M-2, j, k]
                    udy = uxy - (u[M-1, j-1, k] + u[M-1, j+1, k])/2
                    
                    utxy = ut[M-1, j, k]
                    utdx = utxy - ut[M-2, j, k]
                    utdy = utxy - (ut[M-1, j-1, k] + ut[M-1, j+1, k])/2
                    
                    out[M-1, j, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd +  im_convo[M-1, j, k]
                    
    # First column
    with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for k in range(3):
                    uxy = u[i, 0, k]
                    udx = uxy - u[i, 1, k]
                    udy = uxy - (u[i-1, 0, k] + u[i+1, 0, k])/2
                    
                    utxy = ut[i, 0, k]
                    utdx = utxy - ut[i, 1, k]
                    utdy = utxy - (ut[i-1, 0, k] + ut[i+1, 0, k])/2
                    
                    out[i, 0, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd +  im_convo[i, 0, k]             
                    
    # Last column
    with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for k in range(3):
                    uxy = u[i, N-1, k]
                    udx = uxy - u[i, N-2, k]
                    udy = uxy - (u[i-1, N-1, k] + u[i+1, N-1, k])/2
                    
                    utxy = ut[i, N-1, k]
                    utdx = utxy - ut[i, N-2, k]
                    utdy = utxy - (ut[i-1, N-1, k] + ut[i+1, N-1, k])/2
                    
                    out[i, N-1, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd + im_convo[i, N-1, k]
                    
                    
    # North-West corder
    for k in range(3):
        uxy = u[0, 0, k]
        udx = uxy - u[0, 1, k]
        udy = uxy - u[1, 0, k]
        
        utxy = ut[0, 0, k]
        utdx = utxy - ut[0, 1, k]
        utdy = utxy - ut[1, 0, k]
        
        out[0, 0, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd +  im_convo[0, 0, k]
        
        
    # North-East corner
    for k in range(3):
        uxy = u[0, N-1, k]
        udx = uxy - u[0, N-2, k]
        udy = uxy - u[1, N-1, k]
        
        utxy = ut[0, N-1, k]
        utdx = utxy - ut[0, N-2, k]
        utdy = utxy - ut[1, N-1, k]
        
        out[0, N-1, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd + im_convo[0, N-1, k]
                    
                    
    # South-East
    for k in range(3):
        uxy = u[M-1, N-1, k]
        udx = uxy - u[M-1, N-2, k]
        udy = uxy - u[M-2, N-1, k]
    
        utxy = ut[M-1, N-1, k]
        utdx = utxy - ut[M-1, N-2, k]
        utdy = utxy - ut[M-2, N-1, k]
        
        out[M-1, N-1, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd + im_convo[M-1, N-1, k]
        
        
    # South-West
    for k in range(3):
        uxy = u[M-1, 0, k]
        udx = uxy - u[M-2, 0, k]
        udy = uxy - u[M-1, 1, k]
    
        utxy = ut[M-1, 0, k]
        utdx = utxy - ut[M-2, 0, k]
        utdy = utxy - ut[M-1, 1, k]
        
        out[M-1, 0, k] = (udx + udy) / (fabsf(udx) + fabsf(udy) + epsilon) / (fabsf(utdx) + fabsf(utdy) + epsilon + tau) / 2 / lambd + im_convo[M-1, 0, k]


cdef inline void rotate_180(float[:, :, :] array, int M, int N, float[:, :, :] out) nogil:
    """Rotate an array by 2×90° around its center"""

    cdef:
        int i, j, k

    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    out[i, N-1 -j, k] = array[M - i - 1, j, k]
    

cdef void _richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                              float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, float epsilon, int neighbours, int blind=True):
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
    """


    cdef float dtpsf

    cdef int u_M = u.shape[0]
    cdef int u_N = u.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gradk = np.empty_like(psf, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] ut = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gradu = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] im_convo = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] error = np.empty_like(image, dtype=DTYPE)

    cdef float dt[3]
    cdef float max_img[3]
    cdef float max_grad_u[3]
    cdef float max_grad_psf[3]
    cdef float stop = 1e14
    cdef float stop_previous = 1e15
    cdef float stop_2 = 1e14
    cdef float stop_previous_2 = 1e15
    cdef float max_float
    
    # Once convergence is detected on the PSF, the stop_flag allows it to perform another last round to be sure
    cdef int stop_flag = False
    cdef int convergence_flag = False


    # Temporary buffers for intermediate computations - we declare them here to avoid Python calls in sub-routines
    cdef float[:, :] psf_temp_buffer_2D = cvarray(shape=(MK, MK), itemsize=sizeof(float), format="f")
    cdef float[:, :] u_temp_buffer_2D = cvarray(shape=(u_M, u_N), itemsize=sizeof(float), format="f")

    cdef float[:, :, :] psf_rotated = cvarray(shape=(MK, MK, 3), itemsize=sizeof(float), format="f")
    rotate_180(psf, MK, MK, psf_rotated)
    cdef float[:, :, :] u_rotated = cvarray(shape=(u_M, u_N, 3), itemsize=sizeof(float), format="f")

    print("The program is profiling your system to optimize its performance. This can take some time.")
    cdef convolve FFT_valid = convolve(u_M, u_N, MK, MK, "valid")
    print("Optimization 1/3 done !")
    cdef convolve FFT_full = convolve(M, N, MK, MK, "full")
    print("Optimization 2/3 done !")
    cdef convolve FFT_kern_valid = convolve(u_M, u_N, M, N, "valid")
    print("The profiling is done ! Moving on to the next step…")

    cdef int it, itt, i, j, k

    it = 0
    
    # This problem is supposed to be convex so, as soon as the error increases, we shut the solver down because we won't do better
    while it < iterations and not stop_flag:
        # Prepare the deconvolution
        ut[:] = u.copy()
        itt = 0

        stop_previous = stop
        stop_previous_2 = stop_2

        while itt < 5:

            gradu = np.zeros((u_M, u_N, 3), dtype=DTYPE)

            # Compute the new image
            for chan in range(3):
                error[..., chan] = FFT_valid(u[..., chan], psf[..., chan])

            with nogil, parallel(num_threads=CPU):
                for i in prange(M):
                    for k in range(3):
                        for j in range(N):
                            error[i, j, k] -= image[i, j, k]

            for chan in range(3):
                im_convo[..., chan] = FFT_full(error[..., chan], psf_rotated[..., chan])
                            
            # Compute the ratio of the norm of Total Variation between the current major and minor deblured images
            gradTVEM(u, ut, epsilon, tau, im_convo, lambd, gradu, max_grad_u, u_M, u_N, neighbours)
            
            for k in range(3):
                dt[k] = step_factor * (np.amax(u[..., k]) + 1/(u_M * u_N)) / (max_grad_u[k] + float(1e-31))

            with nogil, parallel(num_threads=CPU):
                for i in prange(u_M):
                    for k in range(3):
                        for j in range(u_N):
                            u[i, j, k] = u[i, j, k] - dt[k] * gradu[i, j, k]

            if blind: 
                # PSF update
                for chan in range(C):
                    error[..., chan] = FFT_valid(u[..., chan], psf[..., chan])

                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                error[i, j, k] -= image[i, j, k]

                rotate_180(u, u_M, u_N, u_rotated)

                for chan in range(C):
                    gradk[..., chan] = FFT_kern_valid(u_rotated[..., chan], error[..., chan])

                #amax(psf, MK, MK, max_img)
                #amax_abs(gradk, MK, MK, max_grad_psf)

                dtpsf = step_factor * (np.amax(psf) + 1/(u_M * u_N)) / (np.amax(np.abs(gradk)) + float(1e-31))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(MK):
                        for k in range(3):
                            for j in range(MK):
                                psf[i, j, k] -= dtpsf * gradk[i, j, k]

                _normalize_kernel(psf, MK)

                rotate_180(psf, MK, MK, psf_rotated)

            # Update loop variables
            itt += 1
            it += 1
            lambd *= 1.001
            
            
        # Convergence analysis
        # As images statistics show the cost function decreases only after 100 iterations and converge around 1000, we enforce 20 iterations no matter what 
        stop = norm_L2_3D(gradu, u_M, u_N)/(u_M*u_N)
        
        if stop_previous < stop and convergence_flag:
            # Second round where convergence is met : stop the loop after this round
            stop_flag = True
        
        if stop_previous < stop:
            # First round where convergence is met : raise a flag, do nothing
            convergence_flag = True
            
            
        if blind:
            stop_2 = norm_L2_3D(gradk, MK, MK)/(MK**2)
            
            if stop_previous_2 < stop_2 and convergence_flag:
                # Second round where convergence is met : stop the loop after this round
                stop_flag = True
            
            if stop_previous_2 < stop_2:
                # First round where convergence is met : raise a flag, do nothing
                convergence_flag = True
                

        print("%i iterations completed" % it)

    if blind:
        print("Convergence at %.3f (PSF) - %i iterations." %(stop_2, it))

        if stop_2 > 1:
            print("This blur estimation is likely to be wrong. Increase the confidence, the bias or reduce the blur size.")
    else:
        print("Convergence at %.3f (image) - %i iterations." %(stop, it))

    if it == iterations:
        print("You did not reach a solution inside the number of iterations you set. Allow more iterations or increase the coarseness.")
    else:
        if stop > 1 and blind == False:
            print("Your solution is likely to be blurred. Increase the bias or the confidence, decrease the bias, or move/increase the mask.")


def richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                       float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, float epsilon, int neighbours, int blind=True):
    # Expose the Cython function to Python
    _richardson_lucy_MM(image, u, psf, tau, M, N, C, MK, iterations, step_factor, lambd, epsilon, neighbours, blind=blind)

