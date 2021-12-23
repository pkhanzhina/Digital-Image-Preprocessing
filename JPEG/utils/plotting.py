import matplotlib.pyplot as plt
import numpy as np


def plot_img(img, cmap=None, title=None):
    img = img.astype(np.uint8)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
