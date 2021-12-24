import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


def plot(img, cmap=None, label=None):
    plt.figure(figsize=(18, 10))
    plt.imshow(img, cmap=cmap)
    if label is not None:
        plt.title(label)
    plt.axis('off')
    plt.show()


def rotate_image(image, angle):
    img_pil = Image.fromarray(image)
    return np.asarray(img_pil.rotate(angle, expand=1))
