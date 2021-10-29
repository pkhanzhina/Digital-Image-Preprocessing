import cv2
import numpy as np
from binarization.utils.utils import integral_image

R = 128


def saulova(img, k=0.2, win=31):
    img = np.int64(img)

    I = integral_image(img)
    I_sq = integral_image(img ** 2)

    thresholds = np.zeros_like(img)
    H, W = img.shape[:2]

    half_win = win // 2

    for x in range(0, H):
        for y in range(0, W):
            x1 = max(x - half_win, 0)
            y1 = max(y - half_win, 0)
            x2 = min(x + half_win + 1, H)
            y2 = min(y + half_win + 1, W)

            S1 = I[x1, y1] + I[x2, y2] - I[x1, y2] - I[x2, y1]
            S2 = I_sq[x1, y1] + I_sq[x2, y2] - I_sq[x1, y2] - I_sq[x2, y1]

            n = (x2 - x1) * (y2 - y1)
            # n = win * win
            mean = S1 / n
            std = (S2 - S1 * S1 / n) / n
            _std = np.var(img[x1:x2, y1:y2])

            t = mean * (1 + k * (std ** 0.5 / R - 1))
            thresholds[x, y] = t
    return (img >= thresholds) * 255
