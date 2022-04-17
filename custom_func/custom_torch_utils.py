import torch
import torch.nn
import torch.nn.functional as f
import torchvision

def gaussian_1d_weights(sigma, sym=True):
    kernel_size = int(((sigma - 0.8) / 0.3 + 1) / 0.5 + 1) # default kernel size
    odd = kernel_size % 2
    if not sym and not odd:
        kernel_size += 1
    n = torch.arange(0, kernel_size) - (kernel_size - 1.0) / 2.0
    sig2 = 2 * sigma * sigma
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    w /= w.mean() # 1-normalize
    return w

def gaussian_1d(arr, sigma):
    # (..., h, w)
    w = gaussian_1d_weights(sigma)
    
    print(w.shape)

def gaussian_2d(arr, sigma):
    # (n, h, w)
    n, h, w = arr.shape
    g1d = gaussian_1d_weights(sigma).float().to(arr.device)
    g2d = torch.outer(g1d, g1d)
    #g2d = (g2d - torch.min(g2d.flatten())) / ((torch.max(g2d.flatten()) - torch.min(g2d.flatten())))
    #torchvision.utils.save_image(g2d.unsqueeze(0).unsqueeze(0), "gaussian.png")
    #exit()
    g1d /= torch.sqrt(g2d.sum())
    arr = arr.reshape(n * h, 1, w)
    g1_row = g1d.reshape(1, 1, -1)
    g1_col = g1d.reshape(1, 1, -1)
    # convole on h dimension
    arr = f.conv1d(arr, g1_row, padding="same")
    arr = arr.reshape(n, h, w)
    arr = torch.permute(arr, (0, 2, 1)) # (n, w, h)
    arr = arr.reshape(n * w, 1, h)
    arr = f.conv1d(arr, g1_col, padding="same")
    arr = arr.reshape(n, w, h)
    arr = torch.permute(arr, (0, 2, 1))
    return arr


if __name__ == "__main__":
    arr = torch.randn(5, 100, 200)
    gaussian_2d(arr, 10)
    
