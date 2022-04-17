import os
import io
from PIL import Image
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.ndimage
from skimage.transform import resize
import torch
import torchvision
import torch.nn.functional as F
import time


from custom_func.integration import surface_integration
from custom_func.custom_torch_utils import gaussian_2d
from custom_func.utils import *

gpu_id = 0

def normalize(arr):
    _min = torch.min(arr.flatten())
    _max = torch.max(arr.flatten())
    return (arr - _min) / (_max - _min)

def solve_svd(I):
    print(I.shape)
    # SVD on I using I = L' dot B model
    # I.shape = (N, P) (luminance matrix)
    # L.shape = (3, N), L'.shape = (N, 3) (light matrix)
    # B.shape = (3, P) (pseudo-normal matrix) = normal times albedo (scaling value)

    rank = 3
    U, S, VH = torch.linalg.svd(I, full_matrices=False)
    L_t = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
    B = torch.diag(torch.sqrt(S[:rank])) @ VH[:rank, :] # Pseudo-normal matrix
    L = L_t.T # light matrix (3, N)
    
    return B.float(), L.float()

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
    
    B_blurred = gaussian_2d(B, sigma)
    B_gradx = F.pad(torch.diff(B_blurred, dim=2), (0, 1, 0, 0, 0, 0)).to(B.device)
    B_grady = F.pad(torch.diff(B_blurred, dim=1), (0, 0, 0, 1, 0, 0)).to(B.device)

    B_gradx = B_gradx.reshape(B.shape[0], -1) # (3, H*W)
    B_grady = B_grady.reshape(B.shape[0], -1) # (3, H*W)

    B = B.reshape(B.shape[0], -1) # (3, H*W)
    
    A1 = B[0] * B_gradx[1] - B[1] * B_gradx[0]
    A2 = B[0] * B_gradx[2] - B[2] * B_gradx[0]
    A3 = B[1] * B_gradx[2] - B[2] * B_gradx[1]
    A4 = -B[0] * B_grady[1] + B[1] * B_grady[0]
    A5 = -B[0] * B_grady[2] + B[2] * B_grady[0]
    A6 = -B[1] * B_grady[2] + B[2] * B_grady[1]
    A = torch.stack([A1, A2, A3, A4, A5, A6]).T

    U, S, VH = torch.linalg.svd(A, full_matrices=False)
    # which one is x?
    x = VH[-1] # right singular vector corresponding to smallest singular value
    D = torch.tensor([
        [-x[2], x[5], 1],
        [x[1], -x[4], 0],
        [-x[0], x[3], 0]
    ]).float().to(B.device)
    B = torch.linalg.inv(D) @ B
    L = D.T @ L
    return B, L

def get_A_N_from_B(B):
    # Note that normals may face "inwards"
    A = torch.norm(B, p=2, dim=0)
    N = B / A.unsqueeze(0) # Normal (unit vector)
    return A, N

def normalize_A_N_Z(A, N, Z):
    A = A.reshape(h, w)
    A_normalized = normalize(A) # for visualization

    N = N.reshape(-1, h, w)
    N_normalized = (N + 1) / 2 # for visualization
    N_normalized = torch.permute(N_normalized, (1, 2, 0))

    Z_normalized = normalize(Z)
    return A_normalized, N_normalized, Z_normalized

def solve_photometric_stereo(I, h, w, gaussian_sigma=10, integration_mode="poisson", optimize_gbr=True, flip_gbr=False):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        I = I.cuda()

    B, L = solve_svd(I)

    B, L = integratibility_normalization(B, L, h, w, gaussian_sigma) # play with different sigma
    # Do some processing on B, L (resolving GBR ambiguity)
    B, L, G = optimize_albedos(B, L, optimize_gbr)
    if flip_gbr:
        GBR_flip = get_gbr(0, 0, -1, B.device).float()
        B = GBR_flip @ B
        L = torch.linalg.inv(GBR_flip).T @ L


    A, N = get_A_N_from_B(B)

    Z = surface_integration(N.detach().cpu().numpy(), h, w, integration_mode) # poisson integration is bad

    return B, L, A, N, Z, G

def get_gbr(m, v, l, device):
    G = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [m, v, l]
    ]).float().to(device)
    return G

def compute_albedo_entropy(A, bins=256):
    cnt, _ = torch.histogram(A.cpu(), bins=bins)
    p = cnt / cnt.sum()
    logp = torch.log(p + 1e-10)
    entropy = -(p * logp).sum() # add small value for numerical stability
    return entropy.item() # return scalar

def optimize_albedos(B, L, optimize_gbr):
    if optimize_gbr is not None:
        if optimize_gbr == "coarse_to_fine":
            B, L, G = optimize_albedos_coarse_to_fine(B, L) # B = G^(-T) @ B
        elif optimize_gbr == "brute_force":
            B, L, G = optimize_albedos_brute_force(B, L)
    else:
        G = torch.eye(3) # GBR is just identity
    return B, L, G

# need to partition very small
def optimize_albedos_brute_force(B, L, t=20, m_range=(-5, 5), v_range=(-5, 5), l_range=(0, 5)):
    ms = torch.linspace(m_range[0], m_range[1], t).to(B.device)
    vs = torch.linspace(v_range[0], v_range[1], t).to(B.device)
    ls = torch.linspace(l_range[0]+1e-8, l_range[1], t).to(B.device) # prevent singular value
    m_best, v_best, l_best = None, None, None
    min_entropy = np.inf
    i = 0
    for m in ms:
        for v in vs:
            for l in ls:
                G = get_gbr(m, v, l, B.device)
                B_gbr = torch.linalg.inv(G).T @ B # scale by GBR transform
                A_gbr, N_gbr = get_A_N_from_B(B_gbr)
                entropy = compute_albedo_entropy(A_gbr)
                #print(f"Iter {i}: {entropy}")
                #print(i)
                if entropy < min_entropy:
                    print(f"Iter {i}: {entropy}")
                    min_entropy = entropy
                    m_best, v_best, l_best = m, v, l
                i += 1

    G = get_gbr(m_best, v_best, l_best, B.device)
    #print(np.linalg.inv(G).T)
    #exit()
    B = torch.linalg.inv(G).T @ B # scale by GBR transfo rm
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


config = {
    "cat": {
        "sigma": 10,
        "integration": "frankot",
        "flip_gbr": True
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

if __name___ == "__main__":
    dataset = "frog"
    optimize_gbr = "brute_force"
    data_folder = f"data/{dataset}"
    I, (h, w) = read_images_from_folder(data_folder, None)

    I = torch.tensor(I).float()

    time_start = time.time()
    B, L, A, N, Z, G = solve_photometric_stereo(I, h, w, 
        config[dataset]["sigma"], config[dataset]["integration"], optimize_gbr=optimize_gbr, flip_gbr=config[dataset]["flip_gbr"])
    time_spent = time.time() - time_start
    print("Time elapsed:", time_spent)

    img = plot_surface(Z, title="optimized", dataset=dataset)
    img.save(f"./results/torch/optimized_{dataset}.png")

    B, L, A, N, Z, G = solve_photometric_stereo(I, h, w, 
        config[dataset]["sigma"], config[dataset]["integration"], optimize_gbr=None, flip_gbr=config[dataset]["flip_gbr"])
    img = plot_surface(Z, title="not optimized", dataset=dataset)
    img.save(f"./results/torch/not_optimized_{dataset}.png")

    A_normalized, N_normalized, Z_normalized = normalize_A_N_Z(A, N, -torch.tensor(Z))

    #plt.imshow(Z_normalized, cmap="gray")
    #plt.show()
    #save_image(normalize(I[0].reshape(h, w)), "orig.png")

    save_image(A_normalized.cpu().numpy(), "./results/torch/albedo.png")
    save_image(N_normalized.cpu().numpy(), "./results/torch/normal.png")
    save_image(Z_normalized.cpu().numpy(), "./results/torch/depth.png")
    #generate_relighting_seqeunce(B, h, w, "fixZ", 50, 10, "relight.mp4", loop=2)

