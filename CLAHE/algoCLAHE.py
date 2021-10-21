import numpy as np
import cv2
from CLAHE.utils.utils import processing_zone


class CLAHE:
    def __init__(self, nrof_rows, nrof_cols, clip_limit=None):
        self.nrof_rows = nrof_rows
        self.nrof_cols = nrof_cols
        self.clipping_limit = clip_limit

    def apply(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        grid_h = np.linspace(0, h, self.nrof_rows, dtype=int)
        grid_w = np.linspace(0, w, self.nrof_cols, dtype=int)
        control_coords_x = np.hstack(([grid_h[0]], (grid_h[:-1] + grid_h[1:]) // 2, [grid_h[-1]]))
        control_coords_y = np.hstack(([grid_w[0]], (grid_w[:-1] + grid_w[1:]) // 2, [grid_w[-1]]))

        cdfs = {}
        bins = np.arange(0, 257)
        for i in range(0, len(control_coords_x)):
            for j in range(0, len(control_coords_y)):
                if i - 1 >= 0 and i < self.nrof_rows and j - 1 >= 0 and j < self.nrof_cols:
                    xt, yt = grid_h[i - 1], grid_w[j - 1]
                    xb, yb = grid_h[i], grid_w[j]
                    counts, bins = np.histogram(img[xt:xb, yt:yb].flatten(), bins=bins)
                    if self.clipping_limit:
                        counts = self.clipping_hist(counts)
                    counts = counts / np.sum(counts)
                    cdf = np.cumsum(counts) * 255
                else:
                    cdf = None
                cdfs[i, j] = cdf

        contrast_img = np.zeros_like(img)
        for i in range(0, len(control_coords_x) - 1):
            for j in range(0, len(control_coords_y) - 1):
                x1, x3 = control_coords_x[i], control_coords_x[i + 1]
                y1, y3 = control_coords_y[j], control_coords_y[j + 1]
                cdf1, cdf2, cdf3, cdf4 = cdfs[i, j], cdfs[i, j + 1], cdfs[i + 1, j + 1], cdfs[i + 1, j]
                contrast_img[x1:x3, y1:y3] = processing_zone(img=img[x1:x3, y1:y3],
                                                             control_points=[x3 - x1, y3 - y1],
                                                             cdfs=[cdf1, cdf2, cdf3, cdf4])
        return contrast_img

    def clipping_hist(self, hist):
        top, bottom = self.clipping_limit, 0
        R, C = 255, self.clipping_limit
        while top - bottom > 1:
            middle = (top + bottom) / 2
            S = np.sum(np.maximum(hist - middle, 0))
            if S > (C - middle) * R:
                top = middle
            else:
                bottom = middle
        L = np.sum(np.maximum(hist - bottom, 0)) / R
        clipping_cdf = np.zeros_like(hist)

        ind = hist < bottom
        clipping_cdf[ind] = hist[ind] + L
        clipping_cdf[~ind] = C
        return clipping_cdf
