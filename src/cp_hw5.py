#!/usr/bin/python
""" This is a module for photometric stereo homework (15-463/663/862, Computational Photography, Fall 2020, CMU).

You can import necessary functions into your code as follows:
from cp_hw5 import integrate_poisson, integrate_frankot, load_sources
"""

import numpy as np
from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

def integrate_poisson(zx, zy):
    """Least squares solution
    Poisson Reconstruction Using Neumann boundary conditions
    Input gx and gy
    gradients at boundary are assumed to be zero
    Output : reconstruction
    """
    H, W = zx.shape

    zx[:, -1] = 0
    zy[-1, :] = 0

    # pad
    zx = np.pad(zx, ((1,1), (1,1)))
    zy = np.pad(zy, ((1,1), (1,1)))

    gxx = np.zeros_like(zx)
    gyy = np.zeros_like(zx)

    # Laplacian
    gyy[1:H+2, 0:W+1] = zy[1:H+2, 0:W+1] - zy[0:H+1, 0:W+1]
    gxx[0:H+1, 1:W+2] = zx[0:H+1, 1:W+2] - zx[0:H+1, 0:W+1]
    f = gxx +gyy
    f = f[1:-1, 1:-1]

    # compute eigen values
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    denom = (2*np.cos(np.pi*x/W)-2) + (2*np.cos(np.pi*y/H)-2)

    # compute cosine transform
    fcos = dct2(f)

    # divide. 1st element of denom will be zero and 1st element of fcos and
    # after division should also be zero; so divided rest of the elements
    denom[0,0] = 1
    fcos[0,0] = 0
    fcos = fcos/denom

    # compute inverse dct2
    Z = idct2(fcos)
    return Z

def integrate_frankot(zx, zy):
    """ Integration using the Frankot-Chellappa algorithm.
    Input zx and zy
    Output : reconstruction
    """
    H, W = zx.shape

    # complain if P or Q are too big
    if (H > 512) or (W > 512):
        raise Exception("Input array too big. Choose a smaller window.")

    # pad the input array to 512x512
    nrows = 2**9
    ncols = 2**9

    # compute Fourier coefficients
    Zx = np.fft.fft2(zx, (nrows, ncols))
    Zy = np.fft.fft2(zy, (nrows, ncols))
    H2 = nrows
    W2 = ncols

    Zx = Zx.flatten()
    Zy = Zy.flatten()

    # compute repeated frequency vectors (See Chellapa paper)
    Wx = np.tile(2*np.pi/H2*np.hstack((np.arange(0,H2/2+1), np.arange(-H2/2+1,0))).reshape(-1,1), (W2, 1)).flatten()
    Wy = np.kron(2*np.pi/W2*np.hstack((np.arange(0,W2/2+1), np.arange(-W2/2+1,0))).reshape(-1,1), np.ones((H2,1))).flatten()

    # compute transform of least squares closest integrable surface
    #    remove first column because it's all zeros (then add C[0]=0)
    C = (-1j*Wx[1:]*Zx[1:] - 1j*Wy[1:]*Zy[1:])/(Wx[1:]**2+Wy[1:]**2)

    # set DC component of C
    C = np.hstack((0, C))

    # invert transform to get depth of integrable surface
    Z = np.real(np.fft.ifft2(np.reshape(C,(H2,W2))))

    # crop output if there was padding  
    Z = Z[:H,:W]
    return Z



def load_sources():
    S = np.array([[-0.1418,   -0.1804,   -0.9267], \
                  [ 0.1215,   -0.2026,   -0.9717], \
                  [-0.0690,   -0.0345,   -0.8380], \
                  [ 0.0670,   -0.0402,   -0.9772], \
                  [-0.1627,    0.1220,   -0.9790], \
                  [      0,    0.1194,   -0.9648], \
                  [ 0.1478,    0.1209,   -0.9713]])
    return S
