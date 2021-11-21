import cv2
import numpy as np
from scipy.ndimage import median_filter
from JPEG.utils.utils import get_df


def BAG(img, mask=None, AC=33):
    dx, dy = get_df(img)
    if mask is not None:
        dx[mask == 1] = 0
        dy[mask == 1] = 0
    eh = first_BAG(dx, ax=0, AC=AC)
    gh = second_BAG(eh, ax=0, AC=AC)
    ev = first_BAG(dy, ax=1, AC=AC)
    gv = second_BAG(ev, ax=1, AC=AC)
    g = gh + gv
    return anomaly_score(g)


def first_BAG(d, ax, AC):
    kernel = np.ones((AC, 1)) if ax == 0 else np.ones((1, AC))
    es = cv2.filter2D(d, -1, kernel)
    mid = median_filter(es, kernel.shape[::-1])
    return es - mid


def second_BAG(e, ax, AC):
    kernel = np.zeros(AC)
    kernel[np.arange(0, AC, 8)] = 1
    kernel = kernel.reshape((-1, 1)) if ax == 1 else kernel.reshape((1, -1))
    g = median_filter(e, footprint=kernel)
    return g


def block_value(block):
    max_h = np.max(np.sum(block[1:7, 1:7], axis=0))
    max_v = np.max(np.sum(block[1:7, 1:7], axis=1))
    min_h = np.min(np.sum(block[1:7, (0, 7)], axis=0))
    min_v = np.min(np.sum(block[(0, 7), 1:7], axis=1))
    return max_h - min_h + max_v - min_v


def anomaly_score(g):
    m = 8 - g.shape[0] % 8
    n = 8 - g.shape[1] % 8
    padded_g = np.pad(g, ((0, m), (0, n)))
    b = np.zeros((padded_g.shape[0] // 8, padded_g.shape[1] // 8))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            x, y = 8 * i, 8 * j
            b[i, j] = block_value(padded_g[x:x + 8, y:y + 8])
    return b
