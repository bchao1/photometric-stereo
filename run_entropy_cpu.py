import os
import io
from PIL import Image
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.ndimage
from skimage.transform import resize
import time


from lib.integration import surface_integration
from lib.utils import *


def normalize(arr):
    _min = np.min(arr.ravel())
    _max = np.max(arr.ravel())
    return (arr - _min) / (_max - _min)


def solve_svd(I):
    # SVD on I using I = L' dot B model
    # I.shape = (N, P) (luminance matrix)
    # L.shape = (3, N), L'.shape = (N, 3) (light matrix)
    # B.shape = (3, P) (pseudo-normal matrix) = normal times albedo (scaling value)

    rank = 3
    U, S, VH = np.linalg.svd(I, full_matrices=False)
    L_t = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
    B = np.diag(np.sqrt(S[:rank])) @ VH[:rank, :] # Pseudo-normal matrix
    L = L_t.T # light matrix (3, N)
    
    return B, L

def naive_normalization(B, L):
    """
        Normalizes the light matrix L and the pseudonormal matrix B
        so that each light vector is approximately a unit vector (light direction only).
    """

    L_norm_avg = np.linalg.norm(L, ord=2, axis=0).mean()
    # In this case, the Q matrix is a diagonal matrix in the form of L_norm_avg^(-1) * Identity
    L = L / L_norm_avg # light matrix, approx. unit vector (Q @ L)
    B = B * L_norm_avg # rescale Pseudo-normal = Albedo * Normal (Q^(-T) @ B)
    return B, L

def integratibility_normalization(B, L, h, w, sigma=2):

    B = B.reshape(-1, h, w) # (3, h, w)
    B_gradx = np.zeros_like(B)
    B_grady = np.zeros_like(B)
    for i in range(B.shape[0]):
        B_blurred = scipy.ndimage.gaussian_filter(B[i], sigma)
        B_grady[i], B_gradx[i] = np.gradient(B_blurred, axis=(0, 1)) # tricky!
    B_gradx = B_gradx.reshape(B.shape[0], -1) # (3, H*W)
    B_grady = B_grady.reshape(B.shape[0], -1) # (3, H*W)
    B = B.reshape(B.shape[0], -1) # (3, H*W)
    
    A1 = B[0] * B_gradx[1] - B[1] * B_gradx[0]
    A2 = B[0] * B_gradx[2] - B[2] * B_gradx[0]
    A3 = B[1] * B_gradx[2] - B[2] * B_gradx[1]
    A4 = -B[0] * B_grady[1] + B[1] * B_grady[0]
    A5 = -B[0] * B_grady[2] + B[2] * B_grady[0]
    A6 = -B[1] * B_grady[2] + B[2] * B_grady[1]
    A = np.stack([A1, A2, A3, A4, A5, A6]).T

    U, S, VH = np.linalg.svd(A, full_matrices=False)
    # which one is x?
    x = VH[-1] # right singular vector corresponding to smallest singular value
    D = np.array([
        [-x[2], x[5], 1],
        [x[1], -x[4], 0],
        [-x[0], x[3], 0]
    ])
    B = np.linalg.inv(D) @ B
    L = D.T @ L
    return B, L

def get_A_N_from_B(B):
    # Note that normals may face "inwards"
    A = np.linalg.norm(B, ord=2, axis=0)
    #print("A", A)
    N = B / np.expand_dims(A, 0) # Normal (unit vector)
    #print(A)
    #exit()
    return A, N

def normalize_A_N_Z(A, N, Z):
    A = A.reshape(h, w)
    A_normalized = normalize(A) # for visualization

    N = N.reshape(-1, h, w)
    N_normalized = (N + 1) / 2 # for visualization
    N_normalized = np.transpose(N_normalized, (1, 2, 0))

    Z_normalized = normalize(Z)
    return A_normalized, N_normalized, Z_normalized

def solve_photometric_stereo(I, h, w, gaussian_sigma=10, integration_mode="poisson", optimize_gbr=True, flip_gbr=False):
    B, L = solve_svd(I)

    B, L = integratibility_normalization(B, L, h, w, gaussian_sigma) # play with different sigma
    # Do some processing on B, L (resolving GBR ambiguity)
    B, L, G = optimize_albedos(B, L, optimize_gbr)

    if flip_gbr:
        GBR_flip = get_gbr(0, 0, -1)
        B = GBR_flip @ B 
        L = np.linalg.inv(GBR_flip).T @ L

    A, N = get_A_N_from_B(B)
    Z = surface_integration(N, h, w, integration_mode) # poisson integration is bad
    
    return B, L, A, N, Z, G

def get_gbr(m, v, l):
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [m, v, l]
    ])
    return G

def compute_albedo_entropy(A, bins=256):
    cnt, _ = np.histogram(A, bins=bins)
    p = cnt / cnt.sum()
    logp = np.log(p + 1e-10)
    entropy = -(p * logp).sum() # add small value for numerical stability
    return entropy

def optimize_albedos(B, L, optimize_gbr):
    if optimize_gbr is not None:
        if optimize_gbr == "coarse_to_fine":
            B, L, G = optimize_albedos_coarse_to_fine(B, L) # B = G^(-T) @ B
        elif optimize_gbr == "brute_force":
            B, L, G = optimize_albedos_brute_force(B, L)
    else:
        G = np.eye(3) # GBR is just identity
    return B, L, G

# need to partition very small
def optimize_albedos_brute_force(B, L, t=20, m_range=(-5, 5), v_range=(-5, 5), l_range=(0, 5)):
    ms = np.linspace(m_range[0], m_range[1], t)
    vs = np.linspace(v_range[0], v_range[1], t)
    ls = np.linspace(l_range[0]+1e-8, l_range[1], t) # prevent singular value
    m_best, v_best, l_best = None, None, None
    min_entropy = np.inf
    i = 0
    for m in ms:
        for v in vs:
            for l in ls:
                G = get_gbr(m, v, l)
                B_gbr = np.linalg.inv(G).T @ B # scale by GBR transform
                A_gbr, N_gbr = get_A_N_from_B(B_gbr)
                entropy = compute_albedo_entropy(A_gbr)
                #print(f"Iter {i}: {entropy}")
                #print(i)
                if entropy < min_entropy:
                    print(f"Iter {i}: {entropy}")
                    min_entropy = entropy
                    m_best, v_best, l_best = m, v, l
                i += 1

    G = get_gbr(m_best, v_best, l_best)
    #print(np.linalg.inv(G).T)
    #exit()
    B = np.linalg.inv(G).T @ B # scale by GBR transfo rm
    #print(B)
    #exit()
    L = G @ L # L = G @ L. I = L^T @ B = L^T @ G^T @ G^(-T) @ B = L^T @ B
    print(G)
    return B, L, G

def get_best_gbr_centers(B, t, m_range, v_range, l_range):
    m_edges = np.linspace(m_range[0], m_range[1], t + 1)
    v_edges = np.linspace(v_range[0], v_range[1], t + 1)
    l_edges = np.linspace(l_range[0], l_range[1], t + 1)
    m_best_i, v_best_i, l_best_i = None, None, None
    min_entropy = np.inf
    for i in range(t):
        m = (m_edges[i] + m_edges[i + 1]) / 2
        for j in range(t):
            v = (v_edges[j] + v_edges[j + 1]) / 2
            for k in range(t):
                l = (l_edges[k] + l_edges[k + 1]) / 2
                G = get_gbr(m, v, l)
                B = np.linalg.inv(G).T @ B # scale by GBR transform
                A, N = get_A_N_from_B(B)
                entropy = compute_albedo_entropy(A)
                if entropy < min_entropy:
                    min_entropy = entropy
                    m_best_i, v_best_i, l_best_i = i, j, k
    m_range_new = (m_edges[m_best_i], m_edges[m_best_i + 1])
    v_range_new = (v_edges[v_best_i], v_edges[v_best_i + 1])
    l_range_new = (l_edges[l_best_i], l_edges[l_best_i + 1])
    return m_range_new, v_range_new, l_range_new

def optimize_albedos_coarse_to_fine(B, L, t=2, levels=10, m_range=(-5, 5), v_range=(-5, 5), l_range=(0, 5)):
    opt_seq = []
    for level in range(levels):
        print(f"Optimizing albedo level {level} ...")
        m_range, v_range, l_range = get_best_gbr_centers(B, t, m_range, v_range, l_range)
        print(m_range, v_range, l_range)
        exit()
    
    m = (m_range[0] + m_range[1]) / 2
    v = (v_range[0] + v_range[1]) / 2
    l = (l_range[0] + l_range[1]) / 2
    G = get_gbr(m, v, l)

    B = np.linalg.inv(G).T @ B # scale by GBR transform
    L = G @ L # L = G @ L. I = L^T @ B = L^T @ G^T @ G^(-T) @ B = L^T @ B
    return B, L, G

#  [-0.26315789 -0.26315789  2.36842106]]

config = {
    "cat": {
        "sigma": 10,
        "integration": "frankot",
        "flip_gbr": False
    },
    "women": {
        "sigma": 10,
        "integration": "frankot",
        "flip_gbr": False
    },
    "frog": {
        "sigma": 6.5,
        "integration": "frankot",
        "flip_gbr": True
    }
}

dataset = "frog"
optimize_gbr = "brute_force"
data_folder = f"data/{dataset}"
I, (h, w) = read_images_from_folder(data_folder, None)

time_start = time.time()
B, L, A, N, Z, G = solve_photometric_stereo(I, h, w, 
    config[dataset]["sigma"], config[dataset]["integration"], optimize_gbr=optimize_gbr, flip_gbr=config[dataset]["flip_gbr"])
time_spent = time.time() - time_start
print("Time elapsed:", time_spent)
img = plot_surface(Z, title="optimized", dataset=dataset)
img.save(f"./results/numpy/optimized_{dataset}.png")

B, L, A, N, Z, G = solve_photometric_stereo(I, h, w, 
    config[dataset]["sigma"], config[dataset]["integration"], optimize_gbr=None, flip_gbr=config[dataset]["flip_gbr"])
img = plot_surface(Z, title="not optimized", dataset=dataset)
img.save(f"./results/numpy/not_optimized_{dataset}.png")

#A_normalized, N_normalized, Z_normalized = normalize_A_N_Z(A, N, -Z)


#save_image(A_normalized, f"./results/numpy/albedo_{dataset}.png")
#save_image(N_normalized, f"./results/numpy/normal_{dataset}.png")
#save_image(Z_normalized, f"./results/numpy/depth._{dataset}png")

#generate_relighting_seqeunce(B, h, w, "fixZ", 50, 10, "relight.mp4", loop=2)

