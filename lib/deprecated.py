"""
This is deprecated yet useful code removed from other files and backed-up
"""

# Constant vector for the 3D gradient filter
cdef
float
grad_3_dim1[3]
cdef
float
grad_3_dim2[3]
cdef
float
grad_3_dim3[3]
grad_3_dim1[:] = [1 / 9.0, 0, -1 / 9.0]
grad_3_dim2[:] = [1 / 2.0, 2, 1 / 2.0]
grad_3_dim3[:] = [1 / 4.0, 1, 1 / 4.0]

cdef
inline
void
grad3D(float[:, :, :]
u, int
axis, int
M, int
N, float[:, :, :]
out, float[:, :, :]
temp_buffer):
"""
Convolve a 3D image with a separable kernel representing the 2nd order gradient on the 18 neighbouring pixels with an 
efficient approximation as described by [1]

Performs an spatial AND spectral gradient evaluation to remove all finds of noise, as described in [2]

Reference
--------
    [1] https://cdn.intechopen.com/pdfs-wm/39346.pdf
    [2] http://ieeexplore.ieee.org/document/8094858/#full-text-section
    [3] http://www.songho.ca/dsp/convolution/convolution.html
"""

# Initialize the filter vectors
# The whole 3D kernel can be reconstructed by the outer product in the same order

# Set-up the default configuration to evaluate the gradient along the X direction
cdef
float[:]
vect_one
cdef
float[:]
vect_two
cdef
float[:]
vect_three

cdef
int
C = 3
cdef
size_t
i, j, k, nk = 0

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
        for i in prange(1, M - 1):
            for j in range(N):
                for k in range(C):
                    for nk in range(3):
                        out[i, j, k] += vect_one[nk] * u[i - nk + 1, j, k]

    # Columns gradient
    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(1, N - 1):
                for k in range(C):
                    for nk in range(3):
                        temp_buffer[i, j, k] += vect_two[nk] * out[i, j - nk + 1, k]

    # Depth gradients
    with parallel(num_threads=CPU):
        for i in prange(M):
            for j in range(N):
                for k in range(C):
                    # ensure out is clean
                    out[i, j, k] = 0
                    for nk in range(3):
                        out[i, j, k] += vect_three[nk] * temp_buffer[i, j, period_bound(C, k - nk + 1)]

cdef
inline
void
vect_angle(float[:, :, :]
gradx_A, float[:, :, :]
grady_A, float[:, :, :]
gradz_A, int
M, int
N, float[:, :, :]
out):
"""
Compute the arguments of 3 3D vectors at once
"""

cdef:
float
crossx, crossy, crossz, cross, dot
size_t
i, j, k

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

cdef
inline
void
vect_3D_angle(float[:, :, :]
gradx_A, float[:, :, :]
grady_A, float[:, :, :]
gradz_A, float[:, :, :]
gradx_B, float[:, :, :]
grady_B, float[:, :, :]
gradz_B, int
M, int
N, float[:, :, :]
out):
"""
Compute the angle between 2 vectors of 3 dimensions taken on 3 channels at once

http://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

"""

cdef:
float
crossx, crossy, crossz, cross, dot
size_t
i, j, k

with parallel(num_threads=CPU):
    for i in prange(M):
        for k in range(3):
            for j in range(N):
                crossx = grady_A[i, j, k] * gradz_B[i, j, k] - gradz_A[i, j, k] * grady_B[i, j, k]
                crossy = gradz_A[i, j, k] * gradx_B[i, j, k] - gradx_A[i, j, k] * gradz_B[i, j, k]
                crossz = gradx_A[i, j, k] * grady_B[i, j, k] - grady_A[i, j, k] * gradx_B[i, j, k]
                cross = powf(powf(crossx, 2) + powf(crossy, 2) + powf(crossz, 2), .5)
                dot = gradx_A[i, j, k] * gradx_B[i, j, k] + grady_A[i, j, k] * grady_B[i, j, k] + gradz_A[i, j, k] * \
                                                                                                  gradz_B[i, j, k]
                out[i, j, k] = fabsf(atan2f(cross, dot))
