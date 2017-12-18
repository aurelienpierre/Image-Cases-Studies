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


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Activate debug checks for types and values - Set to False in production
cdef int DEBUG = False


cdef inline int check_nan_2D(float[:, :] array, int M, int N) nogil:
    cdef size_t i, j
    cdef int out = 0

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                if isnan(array[i, j]):
                    out += 1

    return out


cdef inline int check_nan_3D(float[:, :, :] array, int M, int N, int C) nogil:
    cdef size_t i, j, k
    cdef int out = 0

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(C):
                    if isnan(array[i, j, k]):
                        out += 1

    return out


cdef inline float norm_L2_2D(float[:, :] array, int M, int N) nogil:
    cdef size_t i, j
    cdef float out = 0

    with nogil:
        for i in range(M):
            for j in range(N):
                out += powf(array[i, j], 2)

        out = powf(out, 0.5)

    return out


cdef inline float norm_L2_3D(float[:, :] dim1, float[:, :]  dim2, int M, int N) nogil:
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


cdef inline float best_param(float[:, :, :] image, float lambd, int M, int N):
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

    # Average the channels to compute the parameters just once
    cdef float[:, :] image_average = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")

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

    # Compute the difference between the averaged image and the mean
    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                image_average[i, j] -= image_mean

    # Compute omega
    cdef float omega = 2 * lambd * norm_L2_2D(image_average, M, N) / (M*N)

    # Compute the gradient
    cdef float [:, :] gradx = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float [:, :] grady = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float [:, :] temp_buffer_2D = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")

    grad2D(image_average, 0, M, N, gradx, temp_buffer_2D)
    grad2D(image_average, 1, M, N, grady, temp_buffer_2D)

    cdef float grad_mean = norm_L2_3D(gradx, grady, M, N) / (M*N)

    # Compute epsilon
    cdef float epsilon = pow(grad_mean / (expf(omega) - 1), 0.5) * 1.01

    if DEBUG:
        print("best_param passed with epsilon = %f and lambd = %f" % (epsilon, lambd))

    return epsilon


cdef inline void _normalize_kernel(float[:, :, :] kern, int MK):
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
        cdef size_t i, j

        with nogil, parallel(num_threads=CPU):
            for i in prange(self.Mfft):
                for j in range(self.Nfft):
                    cfft[i, j] = Afft[i, j] * Bfft[i, j]

        # Run the iFFT, slice and exit
        cdef np.ndarray[DTYPE_t, ndim=2] out = self.ifft_obj(self.C)

        return out[self.offset_Y:self.offset_Y + self.Y, self.offset_X:self.offset_X + self.X]


cdef inline int sym_bound(int num_elem, int current_index) nogil:
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


cdef inline int period_bound(int num_elem, int current_index) nogil:
    """
    Set the index accordingly to loop over an array dimension using a periodic boundary condition.
    This avoids padding the array with replicated data.
    
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


cdef float grad_2_dim1[3]
cdef float grad_2_dim2[3]
grad_2_dim1[:] = [-0.5, 0, 0.5]
grad_2_dim2[:] = [-1/6.0, -2/3.0, -1/6.0]


cdef inline void grad2D(float[:, :] u, int axis, int M, int N, float[:, :] out, float[:, :] temp_buffer_2D):
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
    cdef float[:] vect_one
    cdef float[:] vect_two

    cdef size_t i, j, nk = 0

    if axis == 0:
        # gradient along rows - y
        vect_one = grad_2_dim1
        vect_two = grad_2_dim2

    elif axis == 1:
        # gradient along columns - x
        vect_one = grad_2_dim2
        vect_two = grad_2_dim1

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

    if DEBUG:
        print("grad2D passed")
        print("out z: Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(out), np.amax(out), np.mean(out), np.median(out), np.std(out)))
    if not np.any(out):
        print(M, N)
        raise ValueError("Grad3D is all null")



cdef float grad_3_dim1[3]
cdef float grad_3_dim2[3]
cdef float grad_3_dim3[3]
grad_3_dim1[:] = [1/9.0, 0, -1/9.0]
grad_3_dim2[:] = [1/2.0, 2, 1/2.0]
grad_3_dim3[:] = [1/4.0, 1, 1/4.0]


cdef inline void grad3D(float[:, :, :] u, int axis, int M, int N, float[:, :, :] out, float[:, :, :] temp_buffer):
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
    cdef float[:] vect_one
    cdef float[:] vect_two
    cdef float[:] vect_three

    cdef int C = 3
    cdef size_t i, j, k, nk = 0

    if axis == 0:
        # gradient along rows - y
        vect_one = grad_3_dim1
        vect_two = grad_3_dim2
        vect_three = grad_3_dim3

    elif axis == 1:
        # gradient along columns - x
        vect_one = grad_3_dim2
        vect_two = grad_3_dim1
        vect_three = grad_3_dim3

    elif axis == 2:
        # gradient along depth - z
        vect_one = grad_3_dim3
        vect_two = grad_3_dim2
        vect_three = grad_3_dim1

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

    if DEBUG:
        print("grad3D passed")
        print("out z: Min =  %f - Max = %f - Mean = %f - Median = %f - Std = %f" % (np.amin(out), np.amax(out), np.mean(out), np.median(out), np.std(out)))

        if check_nan_3D(out, M, N, 3) != 0:
            raise ValueError("Grad3D contains NaN")

        if not np.any(out):
            print(M, N)
            raise ValueError("Grad3D is all null")
        



cdef inline void gradTVEM(float[:, :, :] u, float[:, :, :] ut, float epsilon, float tau, float[:, :, :] out, int M, int N):
    # http://ieeexplore.ieee.org/document/8094858/#full-text-section
    # https://cdn.intechopen.com/pdfs-wm/39346.pdf
    # https://www.intechopen.com/books/matlab-a-fundamental-tool-for-scientific-computing-and-engineering-applications-volume-3/convolution-kernel-for-fast-cpu-gpu-computation-of-2d-3d-isotropic-gradients-on-a-square-cubic-latti

    cdef int C = 2

    cdef size_t i, j, k
    cdef float[:, :, :] gradx = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] grady = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] gradz = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float [:, :, :] temp_buffer = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float max_TV_abs[3]
    cdef float max_TV[3]


    grad3D(u, 1, M, N, gradx, temp_buffer)
    grad3D(u, 0, M, N, grady, temp_buffer)

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    out[i, j, k] = fabsf(gradx[i, j, k]) + fabsf(grady[i, j, k]) + epsilon

    if DEBUG:
        if check_nan_3D(out, M, N, 3) != 0:
            raise ValueError("TV u contains NaN")

        if not np.any(out):
            raise ValueError("TV u is all null")

    grad3D(ut, 1, M, N, gradx, temp_buffer)
    grad3D(ut, 0, M, N, grady, temp_buffer)

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    out[i, j, k] /= fabsf(gradx[i, j, k]) + fabsf(grady[i, j, k]) + epsilon + tau


    # Compute the z-gradient of the TV over the channels. True details/edges should have a similar gradTV over the 3 channels,
    # if they don't, we more likely have chroma noise or phase-shift (chromatic aberrations) there so we add an extra
    # penalty for these (pixels; channels)
    grad3D(out, 2, M, N, gradz, temp_buffer)

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    out[i, j, k] = out[i, j, k] + fabsf(gradz[i, j, k]) / 3

    if DEBUG:
        print("gradTVEM passed")

        if check_nan_3D(out, M, N, 3) != 0:
            raise ValueError("TV ut contains NaN")

        if not np.any(out):
            raise ValueError("TV ut is all null")


cdef inline void amax(float[:, :, :] array, int M, int N, float out[3]) nogil:
    """Compute the max of every channel and output a vector"""

    cdef size_t i, j, k
    out[0] = array[0, 0, 0]
    out[1] = array[0, 0, 1]
    out[2] = array[0, 0, 2]

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    if array[i, j, k] > out[k]:
                        out[k] = array[i, j, k]


cdef inline void amax_abs(float[:, :, :] array, int M, int N, float out[3]) nogil:
    """Compute the max of the absolute value of every channel and output a vector"""

    cdef size_t i, j, k
    out[0] = fabsf(array[0, 0, 0])
    out[1] = fabsf(array[0, 0, 1])
    out[2] = fabsf(array[0, 0, 2])

    with nogil, parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    if fabsf(array[i, j, k]) > out[k]:
                        out[k] = fabsf(array[i, j, k])


cdef void _richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                              float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, int blind=True, accelerate=False):
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


    cdef float eps, alpha, max_gradu, abs_gradu

    cdef int u_M = u.shape[0]
    cdef int u_N = u.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=3] gradk = np.empty_like(psf, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] ut
    cdef np.ndarray[DTYPE_t, ndim=3] gradTV = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] gradTVz
    cdef np.ndarray[DTYPE_t, ndim=3] im_convo = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] gradu = np.empty_like(u, dtype=DTYPE)
    cdef float dt[3]
    cdef float max_img[3]
    cdef float max_grad_u[3]
    cdef float max_grad_psf[3]
    cdef float stop = 1e14
    cdef float stop_previous = 1e15
    cdef float stop_2 = 1e14
    cdef float stop_previous_2 = 1e15

    cdef np.ndarray[DTYPE_t, ndim=3] error = np.empty_like(image, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] u_grad_accel = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] psf_grad_accel, psf_previous

    cdef convolve FFT_valid = convolve(u_M, u_N, MK, MK, "valid")
    cdef convolve FFT_full = convolve(M, N, MK, MK, "full")
    cdef convolve FFT_kern_valid = convolve(u_M, u_N, M, N, "valid")

    cdef float k_step[3]
    cdef float u_step[3]
    cdef size_t step, it, itt, i, j, k, chan
    cdef int main_iter


    # Compute the step factor in the gradient descent
    cdef float iter_weight

    """
    if accelerate and MK <= 9:
        # When the blur is small (lens blur) it is safe to accelerate the gradient descent with coarser values at the beginning
        iterations[:] = [int(iter/6), int(iter/3), int(iter/2)]
        u_step[:] = [step_factor*2, step_factor, step_factor/2]
        k_step[:] = [step_factor*2, step_factor, step_factor/2]
    else:
        # For large motion blurs and tricky smaller blurs, we need to use a fine gradient-descent from the beginning
        iterations[0] = iter
        u_step[0] = step_factor
        k_step[0] = step_factor

    if blind:
        main_iter = 3
    else:
        main_iter = 1

    """

    cdef float inner_it = 1
    cdef float outer_it = 1

    print("max %i iterations to perform" % (iterations*5))

    it = 0

    while it < iterations and stop < stop_previous and stop_previous_2 > stop_2:

        ut = u.copy()
        psf_previous = psf.copy()
        itt = 0

        while itt < 5 and stop_previous > stop and stop_previous_2 > stop_2:

            # Compute the minimal eps parameter that won't degenerate the sharp solution into a constant one
            eps = best_param(u, lambd, u_M, u_N)

            stop_previous = stop

            if accelerate:
                # Accelerated version adapted from [6]
                iter_weight = (itt - 1) / (itt + 2)

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for k in range(3):
                            for j in range(u_N):
                                u_grad_accel[i, j, k] = u[i, j, k] - ut[i, j, k]

                amax(u, u_M, u_N, max_img)
                amax_abs(u_grad_accel, u_M, u_N, max_grad_u)

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for k in range(3):
                            for j in range(u_N):
                                u_grad_accel[i, j, k] = u[i, j, k] + u_grad_accel[i, j, k] * iter_weight * max_grad_u[k] / (max_img[k] + float(1e-31))


                # Compute the ratio of the norm of Total Variation between the current major and minor deblured images
                gradTVEM(u_grad_accel, ut, eps, tau, gradTV, u_M, u_N)

                # Compute the new image
                for chan in range(3):
                    error[..., chan] = FFT_valid(u_grad_accel[..., chan], psf[..., chan])

                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                error[i, j, k] -= image[i, j, k]

                for chan in range(3):
                    im_convo[..., chan] = FFT_full(error[..., chan], np.rot90(psf[..., chan], 2))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for k in range(3):
                            for j in range(u_N):
                                gradu[i, j, k] = lambd *  im_convo[i, j, k] + gradTV[i, j, k]


                amax(u_grad_accel, u_M, u_N, max_img)
                amax_abs(gradu, u_M, u_N, max_grad_u)

                stop = max_grad_u[0] + max_grad_u[1] + max_grad_u[2]

                for k in range(3):
                    dt[k] = step_factor * (max_img[k] + 1/(u_M * u_N)) / (max_grad_u[k] + float(1e-31))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for k in range(3):
                            for j in range(u_N):
                                u[i, j, k] = u_grad_accel[i, j, k] - dt[k] * gradu[i, j, k]


            else:
                # Non-accelerated Perrone's version

                # Compute the ratio of the norm of Total Variation between the current major and minor deblured images
                gradTVEM(u, ut, eps, tau, gradTV, u_M, u_N)

                # Compute the new image
                for chan in range(3):
                    error[..., chan] = FFT_valid(u[..., chan], psf[..., chan])

                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                error[i, j, k] -= image[i, j, k]

                for chan in range(3):
                    im_convo[..., chan] = FFT_full(error[..., chan], np.rot90(psf[..., chan], 2))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for k in range(3):
                            for j in range(u_N):
                                gradu[i, j, k] = lambd *  im_convo[i, j, k] + gradTV[i, j, k]


                amax(u, u_M, u_N, max_img)
                amax_abs(gradu, u_M, u_N, max_grad_u)

                stop = max_grad_u[0] + max_grad_u[1] + max_grad_u[2]

                for k in range(3):
                    dt[k] = step_factor * (max_img[k] + 1/(u_M * u_N)) / (max_grad_u[k] + float(1e-31))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(u_M):
                        for k in range(3):
                            for j in range(u_N):
                                u[i, j, k] = u[i, j, k] - dt[k] * gradu[i, j, k]

            if blind:
                # PSF update
                stop_previous_2 = stop_2

                for chan in range(C):
                    error[..., chan] = FFT_valid(u[..., chan], psf[..., chan])

                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                error[i, j, k] -= image[i, j, k]

                for chan in range(C):
                    gradk[..., chan] = FFT_kern_valid(np.rot90(u[..., chan], 2), error[..., chan])

                amax(psf, MK, MK, max_img)
                amax_abs(gradk, MK, MK, max_grad_psf)

                stop_2 = max_grad_psf[0] + max_grad_psf[1] + max_grad_psf[2]

                for k in range(3):
                    dt[k] = step_factor * (max_img[k] + 1/(u_M * u_N)) / (max_grad_psf[k] + float(1e-31))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(MK):
                        for k in range(3):
                            for j in range(MK):
                                psf[i, j, k] -= dt[k] * gradk[i, j, k]

                _normalize_kernel(psf, MK)


            if DEBUG:
                if check_nan_3D(u, u_M, u_N, 3) != 0:
                    raise ValueError("u is NaN")

                if not np.any(u):
                    raise ValueError("u is all null")

                if check_nan_3D(psf, MK, MK, 3) != 0:
                    raise ValueError("psf is NaN")

                if not np.any(psf):
                    raise ValueError("PSF is all null")

            inner_it +=1
            itt += 1

        outer_it += 5
        it += 1

        print("%i iterations completed" % ((it) * (itt)))

    if blind:
        print("- Diverged at %f (PSF) - %i iterations." %(stop_2, (it) * (itt)))
    else:
        print("- Diverged at %f (image) - %i iterations." %(stop, (it) * (itt)))
    #img = plt.imshow(gradTV[..., 0]/np.amax(gradTV[..., 0]))
    #plt.show()


def richardson_lucy_MM(image, u, psf, tau, M, N, C, MK, iterations, step_factor, float lambd, blind=True, accelerate=False):
    # Expose the Cython function to Python
    _richardson_lucy_MM(image, u, psf, tau, M, N, C, MK, iterations, step_factor, lambd, blind=blind, accelerate=accelerate)
