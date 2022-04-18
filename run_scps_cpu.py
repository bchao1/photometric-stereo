import numpy as np

from run_entropy_cpu import * # move shared functions to custom_func

"""
    Self-Calibrating Photometric Stereo, Shi et al. CVPR 2010
"""

dataset = "cat"
data_folder = f"data/{dataset}"
I, (h, w) = read_images_from_folder(data_folder, None) # (K * P)

I = I.T # (P * K)
I_mean_images = I.mean(axis=1, keepdims=True) # (P, ) mean over all images
I_std_images = I.std(axis=1, keepdims=True) # (P, 1)
I_std_product = I_std_images @ I_std_images.T
I_centered_images = I - I_mean_images
print(I_std_product.shape)


