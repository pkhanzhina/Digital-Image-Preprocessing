import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_img(img, label=None, cmap=None):
    plt.figure(figsize=(18, 10))
    plt.imshow(np.uint8(img), cmap=cmap)
    if label is not None:
        plt.title(label)
    plt.axis('off')
    plt.show()


def load_image(path_to_img):
    return cv2.imread(path_to_img)


def gaussian_smoothing(img, kernel_size, sigma):
    return cv2.GaussianBlur(img, kernel_size, sigma)
