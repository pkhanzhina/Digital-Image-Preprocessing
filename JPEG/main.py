import cv2
import numpy as np
from scipy.ndimage import median_filter
from utils.plotting import plot_img


def get_df(img):
    kernel = np.array([[-1, 2, -1]])
    dx = cv2.filter2D(img, -1, kernel)
    dy = cv2.filter2D(img, -1, kernel.T)
    return np.abs(dx), np.abs(dy)


def first_BAG(d, ax, AC=33):
    kernel = np.ones((AC, 1)) if ax == 1 else np.ones((1, AC))
    es = cv2.filter2D(d, -1, kernel)
    mid = median_filter(es, kernel.shape[::-1])
    return es - mid


def second_BAG(e, ax, AC=33):
    kernel = np.zeros(AC)
    kernel[np.arange(0, AC, 8)] = 1
    if ax == 0:
        kernel = kernel.reshape((-1, 1))
    else:
        kernel = kernel.reshape((1, -1))
    g = median_filter(e, footprint=kernel)
    return g
    # h, w = e.shape[:2]
    # padded_e = np.pad(e, ((16, 16), (16, 16)), mode='reflect')
    # _g = np.zeros((h, w, 5))
    # for i, k in enumerate([0, 8, 16, 24, 32]):
    #     if ax == 0:
    #         _g[:, :, i] = padded_e[k:(h + k), 16:-16]
    #     else:
    #         _g[:, :, i] = padded_e[16:-16, k:(w + k)]
    # _g = np.median(_g, axis=-1)


def block_value(block):
    max_h = np.max(np.sum(block[1:7, 1:7], axis=0))
    max_v = np.max(np.sum(block[1:7, 1:7], axis=1))
    min_h = np.min(np.sum(block[1:7, (0, 7)], axis=0))
    min_v = np.min(np.sum(block[(0, 7), 1:7], axis=1))
    return max_h - min_h + max_v - min_v


def anomaly_score(g):
    padded_g = np.pad(g, ((0, 8 - g.shape[0] % 8), (0, 8 - g.shape[1] % 8)))
    b = np.zeros((padded_g.shape[0] // 8, padded_g.shape[1] // 8))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            x, y = 8*i, 8*j
            b[i, j] = block_value(padded_g[x:x+8, y:y+8])
    return b


if __name__ == '__main__':
    img_dir = 'data/planes_forg_2.jpg'
    img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2GRAY).astype(np.float64)
    dx, dy = get_df(img)

    eh = first_BAG(dx, ax=0)
    gh = second_BAG(eh, ax=0)

    ev = first_BAG(dy, ax=1)
    gv = second_BAG(ev, ax=1)

    g = gh + gv
    b = anomaly_score(g)

    plot_img(g, title='g')
    plot_img(b, cmap='gray', title='b')
