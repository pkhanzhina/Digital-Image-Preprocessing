import numpy as np
from binarization.utils.utils import image_histogram


def otsu(img):
    counts, bins = image_histogram(img, 256)
    counts = counts / (img.shape[0] * img.shape[1])
    mean = np.cumsum(counts * bins)
    func = [0]
    nb, no = 0, 1
    for T in range(1, 256):
        nb += counts[T - 1]
        no = 1 - nb
        if nb == 0:
            mu_b = 0
            mu_o = (mean[-1] - mean[T - 1]) / no
        elif no == 0:
            mu_b = mean[T - 1] / nb
            mu_o = 0
        else:
            mu_b = mean[T - 1] / nb
            mu_o = (mean[-1] - mean[T - 1]) / no

        func.append(no * nb * (mu_b - mu_o) ** 2)

    best_T = bins[np.argmax(func)]
    return (img >= best_T) * 255
