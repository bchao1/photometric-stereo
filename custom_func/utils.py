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

def save_image(img, img_path):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(img_path)

def read_image(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    if max(h, w) > 512: # for frankot integration
        if h > w: 
            # set h to 512
            new_h = 512
            new_w = w * new_h / h
        else:
            # set w to 512
            new_w = 512
            new_h = h * new_w / w
        img = resize(img, (int(new_h), int(new_w))) # will cast to (0, 1)
    return img

def get_luminance(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y

def compare_relight(I_recon, I_orig):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(I_recon, cmap="gray")
    ax[1].imshow(I_orig, cmap="gray")
    plt.show()

def plot_surface(Z, dataset, title, show_plot=True):
    from matplotlib.colors import LightSource 
    from mpl_toolkits.mplot3d import Axes3D
    # Z is an HxW array of surface depths
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    # set 3D figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # add a light and shade to the axis for visual effect # (use the ‘-’ sign since our Z-axis points down)
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)
    # display a surface
    # (control surface resolution using rstride and cstride)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(Z))) # same aspect ratio
    
    surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=4, cstride=4)
    if dataset == "women":
        ax.view_init(elev=50, azim=120)
    elif dataset == "cat":
        ax.view_init(elev=50, azim=120)
    elif dataset == "frog":
        ax.view_init(elev=50, azim=120)
    # turn off axis 
    plt.axis('off') 
    plt.title(title)
    
    # save to image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img = Image.open(img_buf)
    if show_plot:
        plt.show() 
    plt.close()
    return img

def relight(l, B, h, w):
    I = (l @ B).reshape(h, w)
    I = np.clip(I, 0, 255)
    I = I.astype(np.uint8)
    return I

def generate_relighting_seqeunce(B, h, w, mode, N, fps, out_path, loop=0):
    I_relight_seq = []
    if mode == "fixZ":
        theta = np.linspace(0, 360, N) / 360 * 2 * np.pi
        for t in theta: 
            l = np.array([np.sqrt(1) * np.cos(t), np.sqrt(1) * np.sin(t), 0])
            I = relight(l, B, h, w)
            I_relight_seq.append(I)
    elif mode == "full":
        theta = np.linspace(0, 360, 2 * N) / 360 * 2 * np.pi
        eta = np.linspace(0, 180, N) / 360 * 2 * np.pi
        for e in eta:
            for t in theta:
                l = np.array([np.sin(e) * np.cos(t), np.sin(e) * np.sin(t), np.cos(e)])
                I = relight(l, B, h, w)
                I_relight_seq.append(I) # ndarray
    
    if out_path.endswith(".gif"):
        I_relight_seq = [Image.fromarray(I) for I in I_relight_seq]
        I_relight_seq[0].save(out_path, save_all=True, 
            append_images=I_relight_seq[1:], duration=1000/fps, loop=0)
    elif out_path.endswith(".mp4"):
        writer = imageio.get_writer(out_path, fps=fps)
        for i in range(loop):
            for img in I_relight_seq:
                writer.append_data(img)
        writer.close()

def read_images_from_folder(data_folder, samples=None):
    I = []
    h, w = None, None
    img_files = os.listdir(data_folder)
    img_files.sort()
    if samples is not None:
        img_files = np.random.choice(img_files, samples, replace=False)
    for img_file in img_files:
        img = read_image(os.path.join(data_folder, img_file))
        h, w = img.shape[0], img.shape[1]
        if len(img.shape) == 3:
            I.append(get_luminance(img).ravel())
        elif len(img.shape) == 2:
            I.append(img.ravel())
    I = np.stack(I) # I.shape = (N, H * W) = (N, P)
    return I, (h, w)