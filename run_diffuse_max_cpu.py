import numpy as np
import cvxpy as cp
import scipy.io

from lib.utils import read_images_from_folder
import matplotlib.pyplot as plt


dataset = "women"
data_folder = f"data/{dataset}"
I, (h, w) = read_images_from_folder(data_folder, None)

A = scipy.io.loadmat("./matlab/A.mat")["A"]
I = I.reshape(-1, h, w)
E = I - A

fig, ax = plt.subplots(1, 2)
ax[0].imshow(I[0], cmap="gray")
ax[1].imshow(A[0], cmap="gray")
plt.show()

