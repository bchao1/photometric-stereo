
from PIL import Image
import numpy as np

for i in range(7):
    Image.open(f"./data/women/input_{i + 1}.tif").save(f"./data/women_png/input_{i + 1}.jpg")