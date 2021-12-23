import numpy as np


def image_histogram(img, nbins=256):
    bins = np.arange(0, nbins)
    hist = np.zeros_like(bins)
    for i in img.flatten():
        hist[i] += 1
    return hist, bins


def integral_image(img):
    img = np.cumsum(np.cumsum(img, axis=1), axis=0)
    h, w = img.shape
    big_img = np.zeros((h + 1, w + 1))
    big_img[1:, 1:] = img
    return big_img


def _integral_image(img, win=None):
    # img = np.pad(img, (win // 2 + 1, win // 2))
    return np.cumsum(np.cumsum(img, axis=1), axis=0)