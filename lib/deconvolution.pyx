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
    int isnan(float)
    float atan2f(float, float)
    float cosf(float)
    float sinf(float)


cdef float PI = 3.141592653589793

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cdef inline float gaussian_weight(float source, float target, float sigma):
    return expf(-powf(source - target, 2) / (2 * powf(sigma, 2) ) ) / (sigma * powf(2 * PI, 0.5))
    

cdef void gaussian_serie(float[:] serie, float average, float std, int length):
    cdef:
        Py_ssize_t i
        
    for i in range(length):
        serie[i] = gaussian_weight(serie[i], average, std)   


def best_param(float[:, :, :] image, float lambd, int M, int N):
    """
    Find the minimum acceptable epsilon to avoid a degenerate constant solution [2]
    Epsilon is the logarithmic norm prior  

    Reference :
    .. [1] https://pdfs.semanticscholar.org/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    .. [2] http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf
    """
    gradient_norm = np.zeros((M, N, 3), dtype=DTYPE)
    TV(image, gradient_norm, M, N, 0., 1)

    # Remember the gradient is not evaluated along the borders
    mean_gradient_norm = np.sum(np.mean(gradient_norm, axis=2)) / ((M-2)*(N-2))
    flat_image = np.mean(image, axis=2)
    omega = 2 * lambd * np.linalg.norm(flat_image - np.mean(flat_image)) / (M*N)
    epsilon = np.sqrt(mean_gradient_norm / (np.exp(omega) - 1))
    
    print("%1.9f, %.1f" % (epsilon, lambd))

    # Beware of the stats used to compute epsilon
    # It consistently gives epsilon = 0 which raises an error and is unexpected.
    
    return epsilon


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


cdef inline void TV(float[:, :, :] u, float[:, :, :] out, int M, int N, float epsilon, int order) nogil:
    
    cdef:
        Py_ssize_t i, j, k
        float dxdy
        float udx_forw, udx_back, udy_forw, udy_back, udxdy_back, udxdy_forw, udydx_back, udydx_forw
        float maximum
        
    ## In this section we treat only the inside of the picture. See the next section for edges and boundaries exceptions
    
    if order == 2:
        # Evaluate the total variation on 4 neighbours : direct directions, with a bilateral approximation
        with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for j in range(1, N-1):
                    for k in range(3):
                        udx_back = u[i-1, j, k] + u[i+1, j, k] + 2 * u[i, j, k]
                        udy_back = u[i, j-1, k] + u[i, j+1, k] + 2 * u[i, j, k]
                        out[i, j, k] = norm_L2(udx_back, udy_back, epsilon)
                        
    elif order == 1:                  
        # Evaluate the total variation on 4 neighbours : direct directions, with a bilateral approximation
        with parallel(num_threads=CPU):
            for i in prange(1, M-1):
                for j in range(1, N-1):
                    for k in range(3):
                        udx_back = u[i, j, k] - u[i-1, j, k] 
                        udy_back = u[i, j, k] - u[i, j-1, k] 

                        udx_forw = -u[i, j, k] + u[i+1, j, k]
                        udy_forw = -u[i, j, k] + u[i, j+1, k]
                        
                        out[i, j, k] = norm_L2(udx_back, udy_back, epsilon) + norm_L2(udx_forw, udy_forw, epsilon) 
                        out[i, j, k] /= 2.
                        
    ## Warning : borders are ignored !!!


cdef inline void diff(float[:, :, :] u, float[:, :, :] out, int M, int N, float epsilon) nogil:
    
    cdef:
        Py_ssize_t i, j, k
        float dxdy
        float udx_forw, udx_back, udy_forw, udy_back, udxdy_back, udxdy_forw, udydx_back, udydx_forw
        float maximum
        
    ## In this section we treat only the inside of the picture. See the next section for edges and boundaries exceptions
                 
    # Evaluate the total variation on 4 neighbours : direct directions, with a bilateral approximation
    with parallel(num_threads=CPU):
        for i in prange(1, M-1):
            for j in range(1, N-1):
                for k in range(3):
                    #udx_back = u[i, j, k] - u[i-1, j, k] 
                    #udy_back = u[i, j, k] - u[i, j-1, k] 

                    #udx_forw = -u[i, j, k] + u[i+1, j, k]
                    #udy_forw = -u[i, j, k] + u[i, j+1, k]
                    
                    udx_back = u[i-1, j, k] + u[i+1, j, k] + 2 * u[i, j, k]
                    udy_back = u[i, j-1, k] + u[i, j+1, k] + 2 * u[i, j, k]
                    
                    out[i, j, k] = (-udx_back - udy_back) / norm_L2(udx_back, udy_back, epsilon)# + (-udx_forw - udy_forw) / norm_L2(udx_forw, udy_forw, epsilon) 
                    #out[i, j, k] /= 2.
                        
    ## Warning : borders are ignored !!!


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
                              float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, float epsilon, int neighbours, int blind=True, int correlation=False):
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
    
    cdef int order = 2
    
    cdef int u_M = u.shape[0]
    cdef int u_N = u.shape[1]
    cdef Py_ssize_t it, itt, i, j, k

    cdef np.ndarray[DTYPE_t, ndim=3] gradk = np.empty((MK, MK, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] psft = np.empty((MK, MK, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] ut = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] gradu = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] im_convo = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] error = np.zeros((M, N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] TV_u = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] TV_ut = np.zeros((u_M, u_N, 3), dtype=DTYPE)
    

    
    # Construct the array of gaussian weights for the autocovariance metric
    cdef np.ndarray[DTYPE_t, ndim=2] residual = np.empty((M, N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] weights = np.zeros((M, N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] width = np.linspace(-1., 1., num=M, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] height = np.linspace(-1., 1., num=N, dtype=DTYPE)
    
    gaussian_serie(width, 0., 1., M)
    gaussian_serie(height, 0., 1., N)
    
    weights = np.sqrt(np.outer(width, height))
    weights /= np.sum(weights)
    
    cdef float dt
    cdef float dtpsf
    cdef float max_img[3]
    cdef float max_grad_u[3]
    cdef float max_grad_psf[3]
    cdef float stop = 1e14
    cdef float stop_previous = 1e15
    cdef float stop_2 = 1e14
    cdef float stop_previous_2 = 1e15
    cdef float max_float
    cdef float M_r, M_r_prev, min_M_r
      
    # Once convergence is detected or error goes out of bounds, the stop_flag is raised
    cdef int stop_flag = False

    # Temporary buffers for intermediate computations - we declare them here to avoid Python calls in sub-routines
    cdef float[:, :, :] psf_rotated = cvarray(shape=(MK, MK, 3), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] u_rotated = cvarray(shape=(u_M, u_N, 3), itemsize=sizeof(float), format="f")
    
    rotate_180(psf, MK, MK, psf_rotated)

    it = 0
    
    ## From Perrone & Favaro : A logaritmic prior for blind deconvolution
    while it < iterations and not stop_flag:
        # Prepare the deconvolution
        ut[:] = u.copy()
        psft[:] = psf.copy()
        itt = 0

        stop_previous = stop


        while itt < 5 and not stop_flag:
        
            # Sythesize the blur
            for chan in range(3):
                error[..., chan] = convolve(u[..., chan], psf[..., chan], mode="valid")

            #convolvebis(u, psf, error, M, N, MK)

            # Compute the residual
            with nogil, parallel(num_threads=CPU):
                for i in prange(M):
                    for j in range(N):
                        for k in range(3):
                            error[i, j, k] -= image[i, j, k]

            for chan in range(3):
                im_convo[..., chan] = convolve(error[..., chan], psf_rotated[..., chan], mode="full")
                
                            
            # Compute the ratio of the epsilon-norm of Total Variation between the current major and minor deblured images
            TV(u, TV_u, u_M, u_N, epsilon, order)
            TV(ut, TV_ut, u_M, u_N, epsilon, order)
            
            if order == 1:
                ## From Perrone & Favaro
                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                if TV_ut[i, j, k] != 0:
                                    gradu[i, j, k] = (TV_u[i, j, k]) / (TV_ut[i, j, k]) / lambd + im_convo[i, j, k] 
                                else:
                                    gradu[i, j, k] = (TV_u[i, j, k]) / (1e-3) / lambd + im_convo[i, j, k] 
            
            elif order ==2:
                ## Adapted from Perrone & Favaro with Lv, Song, Wang & Le : Image restoration with a high-order total variation minimization method
                ## https://ac.els-cdn.com/S0307904X13001832/1-s2.0-S0307904X13001832-main.pdf?_tid=05b19487-7e64-4d36-823e-3391f48b6e6a&acdnat=1530684412_d946ea0d68205a7991c34b3ad60b0dd7
                # Second order TV minimization problem (Perron & Favaro is first order)
                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                if TV_ut[i, j, k] != 0:
                                    gradu[i, j, k] = (TV_u[i, j, k]) / (TV_ut[i, j, k]) / lambd + powf(ut[i, j, k] - u[i, j, k], 2) / lambd / 4. + im_convo[i, j, k] 
                                else:
                                    gradu[i, j, k] = (TV_u[i, j, k]) / (1e-3) / lambd + powf(ut[i, j, k] - u[i, j, k], 2) / lambd / 2. + im_convo[i, j, k] 
            
            # Scale the gradu factor
            dt = step_factor * (u.max() + 1/(u_M * u_N)) / (np.amax(np.abs(gradu)) + float(1e-31))


            with nogil, parallel(num_threads=CPU):
                for i in prange(u_M):
                    for k in range(3):
                        for j in range(u_N):
                            u[i, j, k] = u[i, j, k] - dt * gradu[i, j, k]

            if blind and not stop_flag: 
                # PSF update
                for chan in range(C):
                    error[..., chan] = convolve(u[..., chan], psf[..., chan], mode="valid")
                    
                with nogil, parallel(num_threads=CPU):
                    for i in prange(M):
                        for k in range(3):
                            for j in range(N):
                                error[i, j, k] -= image[i, j, k]

                rotate_180(u, u_M, u_N, u_rotated)

                for chan in range(C):
                    gradk[..., chan] = convolve(u_rotated[..., chan], error[..., chan], mode="valid")

                dtpsf = step_factor/MK * (np.amax(psf) + 1/(u_M * u_N)) / (np.amax(np.abs(gradk)) + float(1e-31))


                with nogil, parallel(num_threads=CPU):
                    for i in prange(MK):
                        for k in range(3):
                            for j in range(MK):
                                psf[i, j, k] -= dtpsf * gradk[i, j, k]
                                
                if correlation:
                    psf = np.dstack((np.mean(psf, axis=2), np.mean(psf, axis=2), np.mean(psf, axis=2)))

                _normalize_kernel(psf, MK)

                rotate_180(psf, MK, MK, psf_rotated)

            # Update loop variables
            itt += 1
            it += 1
            
         
        ### Convergence analysis
        ## Maybe set a PSNR throttle too : http://www.corc.ieor.columbia.edu/reports/techreports/tr-2004-03.pdf
        
        
        ## From Almeida & Figueiredo : New stopping criteria for iterative blimd image deblurring based on residual whiteness measures
        ## http://www.lx.it.pt/~mtf/Almeida_Figueiredo_SSP2011.pdf 
        if it > 5:
            M_r_prev = M_r
            
        # Compute the residual, average and flatten the channels
        with nogil, parallel(num_threads=CPU):
            for i in prange(M):
                for j in range(N):
                
                    residual[i, j] = 0
                    
                    for k in range(3):
                        # Remember u is image padded with MK//2 along each border
                        residual[i, j] += image[i, j, k] - u[i + MK//2 , j + MK//2, k]
                        
                    residual[i, j] /= 3
            
        # Flatten the residual channels
        residual = np.mean(error, axis=2)
        # Center the mean at zero
        residual -= np.mean(residual)
        # Normalize between -1 and 1
        residual = residual / np.amax(np.abs(residual))
        # Autocorrelate the picture : autocovariance
        residual = convolve(residual, np.rot90(residual, 2), mode="same") 
        # Compute the white noise metric
        M_r = - np.sum(weights * residual**2)
        
        if it == 5:
            min_M_r = M_r
        
        if it > 10:
            if blind:
                # Get more conservative on the error tolerance if the PSF is being estimated
                if M_r < M_r_prev:
                    stop_flag = True
                    print("white autocorellation condition met")
                
            else:
                # Get more sloppy on the error if we do a non-blind deconvolution
                if M_r < (1 + tau) * M_r_prev or M_r < (1 + tau) * min_M_r:
                    stop_flag = True
                    print("white autocorellation condition met")
    
    
        ## Adapted from Perrone & Favaro stats
        # log of the energy aka cost function
        stop = np.log(np.linalg.norm(TV_u))

        # Based on the hyper-laplacian distribution of the TV, from Perrone & Favaro
        if stop >= lambd * np.linalg.norm(np.mean(u) - u)**2 + 0.5 * u_M * u_N * np.log(epsilon**2):
            stop_flag = True
            print("statistic condition met")
            
        if it % 20 == 0:
            print("%i iterations completed" % it)
            
    if stop_flag:
        # When one stopping condition has been met, the solutions u and psf have already past degeneration by one step
        # So we retrieve and output the solution from the step before
        
        u[:] = ut.copy()
        psf[:] = psft.copy()

        print("Convergence at function cost %.6f and autocovariance %.6f - %i iterations." %(stop_previous/(M*N)*1000, M_r_prev/(M*N)*1000, it-5))
    else:  
        print("Convergence at function cost %.6f and autocovariance %.6f - %i iterations." %(stop/(M*N)*1000, M_r/(M*N)*1000, it))

    if it == iterations:
        print("You did not reach a solution inside the number of iterations you set. Allow more iterations or increase the coarseness.")
   

def richardson_lucy_MM(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[DTYPE_t, ndim=3] u, np.ndarray[DTYPE_t, ndim=3] psf,
                       float tau, int M, int N, int C, int MK, int iterations, float step_factor, float lambd, float epsilon, int neighbours, int blind=True, int correlation=False):
    # Expose the Cython function to Python
    _richardson_lucy_MM(image, u, psf, tau, M, N, C, MK, iterations, step_factor, lambd, epsilon, neighbours, blind=blind, correlation=correlation)

