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

# Activate debug checks for types and values - Set to False in production
cdef int DEBUG = False

cdef inline float norm_L2_2D(float[:, :] array, int M, int N) nogil:
    """L2 vector norm of a 2D array. Outputs a scalar."""

    cdef size_t i, j
    cdef float out = 0

    for i in range(M):
        for j in range(N):
            out += powf(array[i, j], 2)

    out = powf(out, 0.5)

    return out


cdef inline float norm_L2_2D2(float[:, :] dim1, float[:, :]  dim2, int M, int N) nogil:
    """L2 vector norm of two 2D arrays. Outputs a scalar."""

    cdef size_t i, j
    cdef float out = 0

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                dim1[i, j] = powf((powf(dim1[i, j], 2) + powf(dim2[i, j], 2)), 0.5)

    out = norm_L2_2D(dim1, M, N)

    return out


cdef inline float norm_L2_3D(float [:, :, :] array, int M, int N, float[:, :] temp_buffer_2D) nogil:
    """L2 vector norm of a 3D array. The L2 norm is computed successively on the 3rd axis, then on the resulting 2D array."""

    cdef size_t i, j, k
    cdef float out = 0

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                temp_buffer_2D[i, j] = powf((powf(array[i, j, 0], 2) + powf(array[i, j, 1], 2) + powf(array[i, j, 2], 2)), 0.5)


    out = norm_L2_2D(temp_buffer_2D, M, N)

    return out


cdef inline void amax(float[:, :, :] array, int M, int N, float out[3]) nogil:
    """Compute the max of every channel and output a 3D vector"""

    cdef size_t i, j, k
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

    cdef size_t i, j, k
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

    cdef size_t i, j
    cdef float temp
    cdef float out = fabsf(array[0, 0])

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                temp = fabsf(array[i, j])
                if temp > out:
                    out = temp

    return out


cdef inline float best_param(float[:, :, :] image, float lambd, int M, int N):
    """
    Find the minimum acceptable epsilon to avoid a degenerate constant solution [2]
    Epsilon is the logarithmic norm prior  

    Reference :
    .. [1] https://pdfs.semanticscholar.org/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    .. [2] http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf
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
    cdef int p = 1
    cdef float omega = 2 * lambd * norm_L2_2D(image_average, M, N) / (p*M*N)

    # Compute the gradient
    cdef float [:, :] gradx = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float [:, :] grady = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float [:, :] temp_buffer_2D = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")

    grad2D(image_average, 0, M, N, gradx, temp_buffer_2D)
    grad2D(image_average, 1, M, N, grady, temp_buffer_2D)

    cdef float grad_mean = norm_L2_2D2(gradx, grady, M, N) / (M*N)

    # Compute epsilon
    cdef float epsilon = pow(grad_mean / (expf(omega) - 1), 0.5) * 1.01

    if DEBUG:
        print("best_param passed with epsilon = %f and lambd = %f" % (epsilon, lambd))

    return epsilon


cdef inline void _normalize_kernel(float[:, :, :] kern, int MK):
    """Normalizes a 3D kernel along its 2 first dimensions"""

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


# Constant vector for the gradient filters
cdef :
    float grad_2_dim1[3]
    float grad_2_dim2[3]

grad_2_dim1 = [-0.5, 0, 0.5]
grad_2_dim2 = [-1/6.0, -2/3.0, -1/6.0]


cdef inline void grad2D(float[:, :] u, int axis, int M, int N, float[:, :] out, float[:, :] temp_buffer_2D) nogil:
    """
    Convolve a 2D image with a separable kernel representing the 2nd order 2D gradient on the neighbouring pixels with an 
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
    cdef:
        float *vect_one
        float *vect_two
        size_t i, j, nk

    if axis == 0:
        # gradient along rows - y
        vect_one = grad_2_dim1
        vect_two = grad_2_dim2

    elif axis == 1:
        # gradient along columns - x
        vect_one = grad_2_dim2
        vect_two = grad_2_dim1

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


cdef inline void grad2D_3(float[:, :, :] u, int axis, int M, int N, float[:, :, :] out, float[:, :, :] temp_buffer_3D) nogil:
    """
    Convolve a 3D image with a separable kernel representing the 2nd order 2D gradient on the neighbouring pixels with an 
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
    cdef:
        float *vect_one
        float *vect_two
        size_t i, j, k, nk

    if axis == 0:
        # gradient along rows - y
        vect_one = grad_2_dim1
        vect_two = grad_2_dim2

    elif axis == 1:
        # gradient along columns - x
        vect_one = grad_2_dim2
        vect_two = grad_2_dim1

    # Cleanup the buffers
    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(3):
                    out[i, j, k] = 0
                    temp_buffer_3D[i, j, k] = 0

    # Convolve the filter along the rows
    with parallel(num_threads=CPU):
        for i in prange(1, M-1):
            for j in range(N):
                for k in range(3):
                    for nk in range(3):
                        temp_buffer_3D[i, j, k] += vect_one[nk] * u[i, j, k]

    # Convolve the filter along the columns
    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(1, N-1):
                for k in range(3):
                    for nk in range(3):
                        out[i, j, k] += vect_two[nk] * temp_buffer_3D[i, j, k]


cdef inline void max_3D(float[:, :, :] u, int M, int N, float[:, :] out) nogil:
    """
    Output the maximum value of the 3 RGB channel for each pixel
    :param u: 
    :param M: 
    :param N: 
    :param out: 
    :return: 
    """
    cdef:
        size_t i, j
        float temp

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                if u[i, j, 0] < u[i, j, 1]:
                    temp = u[i, j, 1]
                else:
                    temp = u[i, j, 0]

                if u[i, j, 2] > temp:
                    temp = u[i, j, 2]

                out[i, j] = temp

cdef :
    float grad_3_dim1[3]
    float grad_3_dim2[3]
    float grad_3_dim3[3]

grad_3_dim1[:] = [1/9.0, 0, -1/9.0]
grad_3_dim2[:] = [1/2.0, 2, 1/2.0]
grad_3_dim3[:] = [1/4.0, 1, 1/4.0]


cdef inline void grad3D(float[:, :, :] u, int axis, int M, int N, float[:, :, :] out, float[:, :, :] temp_buffer) nogil:
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
    cdef:
        float *vect_one
        float *vect_two
        float *vect_three

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



cdef inline void vect_arg(float[:, :, :] gradx_A, float[:, :, :] grady_A, float[:, :, :] gradz_A, int M, int N, float[:, :, :] out) nogil:
    """
    Compute the arguments of 3 3D vectors at once
    """

    cdef float crossx, crossy, crossz, cross, dot
    cdef size_t i, j, k

    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(3):
                    crossx = grady_A[i, j, k] - gradz_A[i, j, k]
                    crossy = gradz_A[i, j, k] - gradx_A[i, j, k]
                    crossz = gradx_A[i, j, k] - grady_A[i, j, k]
                    cross = powf(powf(crossx, 2) + powf(crossy, 2) + powf(crossz, 2), .5)
                    dot = gradx_A[i, j, k] + grady_A[i, j, k] + gradz_A[i, j, k]
                    out[i, j, k] = atan2f(cross, dot)


cdef inline void vect_3D_angle(float[:, :, :] gradx_A, float[:, :, :] grady_A, float[:, :, :] gradz_A, float[:, :, :] gradx_B, float[:, :, :] grady_B, float[:, :, :] gradz_B, int M, int N, float[:, :, :] out) nogil:
    """
    Compute the angle between 2 vectors of 3 dimensions taken on 3 channels at once
    
    http://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html
    """

    cdef float crossx, crossy, crossz, cross, dot
    cdef size_t i, j, k

    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    crossx = grady_A[i, j, k] * gradz_B[i, j, k] - gradz_A[i, j, k] * grady_B[i, j, k]
                    crossy = gradz_A[i, j, k] * gradx_B[i, j, k] - gradx_A[i, j, k] * gradz_B[i, j, k]
                    crossz = gradx_A[i, j, k] * grady_B[i, j, k] - grady_A[i, j, k] * gradx_B[i, j, k]
                    cross = powf(powf(crossx, 2) + powf(crossy, 2) + powf(crossz, 2), .5)
                    dot = gradx_A[i, j, k] * gradx_B[i, j, k] + grady_A[i, j, k] * grady_B[i, j, k] + gradz_A[i, j, k] * gradz_B[i, j, k]
                    out[i, j, k] = fabsf(atan2f(cross, dot))


cdef inline void vect_2D_angle(float[:, :, :] gradx_A, float[:, :, :] grady_A, float[:, :, :] gradx_B, float[:, :, :] grady_B, int M, int N, float[:, :, :] out) nogil:
    """
    Compute the angle between 2 vectors of 3 dimensions taken on 3 channels at once
    
    http://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html
    """

    cdef float crossx, crossy, crossz, cross, dot
    cdef size_t i, j, k

    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    crossx = grady_A[i, j, k] * gradx_B[i, j, k] - gradx_A[i, j, k] * grady_B[i, j, k]
                    crossy = gradx_A[i, j, k] * grady_B[i, j, k] - gradx_B[i, j, k] * grady_A[i, j, k]
                    cross = powf(powf(crossx, 2) + powf(crossy, 2), .5)
                    dot = gradx_A[i, j, k] * gradx_B[i, j, k] + grady_A[i, j, k] * grady_B[i, j, k]
                    out[i, j, k] = fabsf(atan2f(cross, dot))


cdef inline void abs_inplace_3D(float [:, :, :] array, int M, int N) nogil:
    cdef size_t i, j, k
    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    array[i, j, k] = fabsf(array[i, j, k])


cdef inline void gradTVEM_denoise(float[:, :, :] u, float[:, :, :] ut, float epsilon, float tau, float lambd, float[:, :] out, int M, int N):
    """
    
    This is the gradient of the Total variation estimation. It has been adapted from the version Perrone & Favaro (2015).
    
    The gradient over the z-axis (RGB channels) has been added and the norm used is L2, allowing to use the argument 
    of the gradient in polar coordinates. Hence, we minimize not only the gradient norm but the angle between the gradient 
    vector of the maximized image and of the minimized one. This improves the convergence by avoiding edges to slip in the 
    wrong direction because we enforce the direction of the gradient.
    
    References :
    --
    .. [1] http://ieeexplore.ieee.org/document/8094858/#full-text-section
    .. [2] https://cdn.intechopen.com/pdfs-wm/39346.pdf
    .. [2] https://www.intechopen.com/books/matlab-a-fundamental-tool-for-scientific-computing-and-engineering-applications-volume-3/convolution-kernel-for-fast-cpu-gpu-computation-of-2d-3d-isotropic-gradients-on-a-square-cubic-latti
    """

    cdef int C = 2

    cdef size_t i, j, k
    cdef float [:, :, :] temp_buffer_3D = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")

    cdef float [:, :, :] theta_u = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    #cdef float [:, :, :] theta_ut = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")

    cdef float[:, :, :] gradx_u = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] grady_u = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] gradz_u = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")

    cdef float[:, :, :] gradx_ut = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] grady_ut = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] gradz_ut = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")

    cdef float[:, :] max_theta = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float[:, :] max_u_x = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float[:, :] max_u_y = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
    cdef float[:, :] max_u_z = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")


    with nogil:
        # compute the gradients
        grad2D_3(u, 1, M, N, gradx_u, temp_buffer_3D)
        grad2D_3(u, 0, M, N, grady_u, temp_buffer_3D)
        grad2D_3(ut, 1, M, N, gradx_ut, temp_buffer_3D)
        grad2D_3(ut, 0, M, N, grady_ut, temp_buffer_3D)

        vect_2D_angle(gradx_u, grady_u, gradx_ut, grady_ut, M, N, theta_u)
        abs_inplace_3D(theta_u, M, N)

        max_3D(theta_u, M, N, max_theta)
        max_3D(gradx_u, M, N, max_u_x)
        max_3D(grady_u, M, N, max_u_y)

        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                    out[i, j] = sinf(max_theta[i, j]) * powf(powf(max_u_x[i, j], 2) + powf(max_u_y[i, j], 2), 0.5)



cdef inline void gradTVEM(float[:, :, :] u, float[:, :, :] ut, float epsilon, float tau, float[:, :] out, int M, int N):
    """
    
    This is the gradient of the Total variation estimation. It has been adapted from the version Perrone & Favaro (2015).
    
    The norm used is Shatten 1 which allows collaboration between channels (see [4]) and gives smoother results.
    
    References :
    --
    .. [1] http://ieeexplore.ieee.org/document/8094858/#full-text-section
    .. [2] https://cdn.intechopen.com/pdfs-wm/39346.pdf
    .. [3] https://www.intechopen.com/books/matlab-a-fundamental-tool-for-scientific-computing-and-engineering-applications-volume-3/convolution-kernel-for-fast-cpu-gpu-computation-of-2d-3d-isotropic-gradients-on-a-square-cubic-latti
    .. [4] https://joandurangrimalt.wordpress.com/research/novel-tv-based-regularization/
    """

    cdef:
        size_t i, j, k
        float [:, :, :] temp_buffer_3D = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
        float[:, :, :] gradx_u = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
        float[:, :, :] grady_u = cvarray(shape=(M, N, 3), itemsize=sizeof(float), format="f")
        float[:, :] max_gradx = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")
        float[:, :] max_grady = cvarray(shape=(M, N), itemsize=sizeof(float), format="f")


    with nogil:
        # Compute the norm of u
        grad2D_3(u, 1, M, N, gradx_u, temp_buffer_3D)
        grad2D_3(u, 0, M, N, grady_u, temp_buffer_3D)
        max_3D(gradx_u, M, N, max_gradx)
        max_3D(grady_u, M, N, max_grady)

        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                    out[i, j] = fabsf(max_gradx[i, j]) + fabsf(max_grady[i, j]) + epsilon

        # Divide by the norm of ut
        grad2D_3(ut, 1, M, N, gradx_u, temp_buffer_3D)
        grad2D_3(ut, 0, M, N, grady_u, temp_buffer_3D)
        max_3D(gradx_u, M, N, max_gradx)
        max_3D(grady_u, M, N, max_grady)

        with parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                    out[i, j] /= fabsf(max_gradx[i, j]) + fabsf(max_grady[i, j]) + epsilon + tau


cdef inline void rotate_180(float[:, :, :] array, int M, int N, float[:, :, :] out) nogil:
    """Rotate an array by 2×90° around its center"""

    cdef:
        size_t i, j, k

    with parallel(num_threads=CPU):
        for i in prange(M):
            for k in range(3):
                for j in range(N):
                    out[i, N-1 -j, k] = array[M - i - 1, j, k]


cdef void _richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                              float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, int denoise, int blind=True):
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


    cdef float eps, iter_weight

    cdef int u_M = u.shape[0]
    cdef int u_N = u.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gradk = np.empty_like(psf, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] ut = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] gradTV = np.empty((u_M, u_N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gradTV3D = np.empty((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] im_convo = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] gradu = np.empty_like(u, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode="c"] u_initial = u.copy()

    cdef float dt
    cdef float max_img[3]
    cdef float max_grad_u[3]
    cdef float max_grad_psf[3]
    cdef float stop = 1e14
    cdef float stop_previous = 1e15
    cdef float stop_2 = 1e14
    cdef float stop_previous_2 = 1e15
    cdef float max_float

    cdef np.ndarray[DTYPE_t, ndim=3] error = np.empty_like(image, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] u_grad_accel = np.empty_like(u, dtype=DTYPE)

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

    cdef size_t it, itt, i, j, k

    it = 0

    # This problem is supposed to be convex so, as soon as the error increases, we shut the solver down because we won't to better
    while it < iterations and stop_previous - stop > 0:# and stop_previous_2 - stop_2 > 0:
        # Prepare the deconvolution
        ut[:] = u.copy()
        itt = 0

        while itt < 5 and stop_previous - stop > 0:# and stop_previous_2 - stop_2 > 0:

            # Compute the minimal eps parameter that won't degenerate the sharp solution into a constant one
            eps = best_param(u, lambd, u_M, u_N)

            stop_previous = stop


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

            # BEGIN Peronne's regular code

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
                im_convo[..., chan] = FFT_full(error[..., chan], psf_rotated[..., chan])

            with nogil, parallel(num_threads=CPU):
                for i in prange(u_M):
                    for k in range(3):
                        for j in range(u_N):
                            gradu[i, j, k] = lambd *  im_convo[i, j, k] + gradTV[i, j]


            stop = norm_L2_3D(gradu, u_M, u_N, u_temp_buffer_2D)/(u_M*u_N)

            dt = step_factor * (np.amax(u) + 1/(u_M * u_N)) / (np.amax(np.abs(gradu)) + float(1e-31))

            with nogil, parallel(num_threads=CPU):
                for i in prange(u_M):
                    for k in range(3):
                        for j in range(u_N):
                            u[i, j, k] = u_grad_accel[i, j, k] - dt * gradu[i, j, k]

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

                rotate_180(u, u_M, u_N, u_rotated)

                for chan in range(C):
                    gradk[..., chan] = FFT_kern_valid(u_rotated[..., chan], error[..., chan])

                #amax(psf, MK, MK, max_img)
                #amax_abs(gradk, MK, MK, max_grad_psf)

                stop_2 = norm_L2_3D(gradk, MK, MK, psf_temp_buffer_2D)/(MK**2)

                dt = step_factor * (np.amax(psf) + 1/(u_M * u_N)) / (np.amax(np.abs(gradk)) + float(1e-31))

                with nogil, parallel(num_threads=CPU):
                    for i in prange(MK):
                        for k in range(3):
                            for j in range(MK):
                                psf[i, j, k] -= dt * gradk[i, j, k]

                _normalize_kernel(psf, MK)

                rotate_180(psf, MK, MK, psf_rotated)

            itt += 1
            it += 1

        lambd *= 1.001

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
                       float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, int denoise, int blind=True):
    # Expose the Cython function to Python
    _richardson_lucy_MM(image, u, psf, tau, M, N, C, MK, iterations, step_factor, lambd, denoise, blind=blind)
