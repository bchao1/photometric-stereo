import numpy as np
import cvxpy as cp
import scipy.io
from scipy.spatial.distance import cdist, euclidean

from lib.utils import read_images_from_folder
from run_entropy_cpu import *
import matplotlib.pyplot as plt
import scipy.ndimage


dataset = "women"
data_folder = f"data/{dataset}"
I, (h, w) = read_images_from_folder(data_folder, None)

#A = scipy.io.loadmat("./data/women_jpg/A.mat")["A"]
I = I.reshape(-1, h, w) # diffuse
#I = A

# 
B, L = solve_svd(I.reshape(-1, h * w))
B, L = integratibility_normalization(B, L, h, w, config[dataset]["sigma"])
#

sigma = 2 # larger sigma, smaller number of LDR peaks
LDR_peaks = np.zeros_like(I)
for i in range(I.shape[0]):
    blurred = scipy.ndimage.gaussian_filter(I[i], sigma)
    lm = scipy.ndimage.filters.maximum_filter(blurred, size=3)
    LDR_peaks[i] = (blurred == lm)

# locations to reject (1 = reject)
repeated_loc = (LDR_peaks.sum(axis=0) >= 2).astype(int)
intensity_threshold = (np.amax(I, axis=(1, 2), keepdims=True) - np.amin(I, axis=(1, 2), keepdims=True)) * 0.5
low_intensity_loc = (I < intensity_threshold).astype(int) 

# Masked out rejected regions
LDR_peaks = LDR_peaks * (1 - repeated_loc) * (1 - low_intensity_loc)

def get_line_segment(l, b):
    u0 = (-(l[1]**2)*b[0] + l[0]*l[1]*b[1] + l[0]*l[2]*b[2]) / (b[2] * (l[0]**2 + l[1]**2))
    v0 = (l[0]*l[1]*b[0] - (l[0]**2)*b[1] + l[1]*l[2]*b[2]) / (b[2] * (l[0]**2 + l[1]**2))
    u1 = -b[0] / b[2]
    v1 = -b[1] / b[2]
    return (u0, v0), (u1, v1)

def get_lambda(alpha, n, l):
    lam = np.sqrt(alpha * (1 - alpha)) * np.abs(l.T @ n) / (np.abs(n[2]) * np.sqrt(l[0]**2 + l[1]**2))
    return lam

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
# 1. geometric median
# 2. RANSAC
gbr_params = []
for i in range(1000):
    sampled_lights = np.random.choice(LDR_peaks.shape[0], 2, replace=False)
    sampled_I = LDR_peaks[sampled_lights]

    LDR_loc1 = np.where(sampled_I[0])
    LDR_loc2 = np.where(sampled_I[1])

    p1 = np.random.choice(LDR_loc1[0]), np.random.choice(LDR_loc1[1]) # 1st image
    p2 = np.random.choice(LDR_loc2[0]), np.random.choice(LDR_loc2[1]) # 2nd image

    #plt.imshow(LDR_peaks[0], cmap="gray")
    #plt.show()

    l1 = L[:, sampled_lights[0]]
    n1 = B.reshape(-1, h, w)[:, p1[0], p1[1]]
    l2 = L[:, sampled_lights[1]]
    n2 = B.reshape(-1, h, w)[:, p2[0], p2[1]]

    (u10, v10), (u11, v11) = get_line_segment(l1, n1)
    (u20, v20), (u21, v21) = get_line_segment(l2, n2)

    a1 = (v11 - v10) / (u11 - u10)
    b1 = (u11*v10 - u10*v11) / (u11 - u10)
    a2 = (v21 - v20) / (u21 - u20)
    b2 = (u21*v20 - u20*v21) / (u21 - u20)

    u = -(b1 - b2) / (a1 - a2)
    v = (a2*b1 - a1*b2) / (a2 - a1)

    alpha1 = (u - u11) / (u10 - u11)
    alpha2 = (u - u21) / (u20 - u21)
    if alpha1 < 0 or alpha1 > 1 or alpha2 < 0 or alpha2 > 1:
        continue
    #print(alpha)
    lam1 = get_lambda(alpha1, n1, l1)
    lam2 = get_lambda(alpha2, n2, l2)
    lam = (lam1 + lam2) * 0.5
    gbr_param = np.array([u, v, lam])
    gbr_params.append(gbr_param)
    #plt.plot(0, lam, "ro")


gbr_params = np.stack(gbr_params)
print(gbr_params.shape)
u_, v_, l_ = geometric_median(gbr_params)
G = get_gbr(u_, v_, l_)

B = np.linalg.inv(G).T @ B # scale by GBR transform
L = G @ L # L = G @ L. I = L^T @ B = L^T @ G^T @ G^(-T) @ B = L^T @ B
if config[dataset]["flip_gbr"]:
    GBR_flip = get_gbr(0, 0, -1)
    B = GBR_flip @ B 
    L = np.linalg.inv(GBR_flip).T @ L

A, N = get_A_N_from_B(B)
Z = surface_integration(N, h, w, config[dataset]["integration"]) # poisson integration is bad
plot_surface(Z, title="optimized", dataset=dataset)

B, L, A, N, Z, G = solve_photometric_stereo(I.reshape(-1, h*w), h, w, 
        config[dataset]["sigma"], config[dataset]["integration"], optimize_gbr=None, flip_gbr=config[dataset]["flip_gbr"])
img = plot_surface(Z, title="not optimized", dataset=dataset)