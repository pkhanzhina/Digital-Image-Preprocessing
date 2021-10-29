import numpy as np
from binarization.utils.utils import integral_image

R = 128


def saulova(img, k=0.2, win=31):
    I = integral_image(img)
    I_sq = integral_image(img ** 2)

    thresholds = np.zeros_like(img)
    H, W = img.shape[:2]
    half_win = win // 2

    for x in range(1, H + 1):
        for y in range(1, W + 1):
            x1 = max(x - half_win, 0)
            y1 = max(y - half_win, 0)
            x2 = min(x + half_win, H)
            y2 = min(y + half_win, W)
            n = (x2 - x1) * (y2 - y1)
            mean = I[x1, y1] + I[x2, y2] - \
                   I[x1, y2] - I[x2, y1]
            sigma = I_sq[x1, y1] + I_sq[x2, y2] - \
                    I_sq[x1, y2] - I_sq[x2, y1]

            mean = mean / n
            std = abs((sigma - (mean * mean)) / n) ** 0.5

            t = mean * (1 + k * (std / R - 1))

            thresholds[x - 1, y - 1] = t
    return (img >= thresholds) * 255
