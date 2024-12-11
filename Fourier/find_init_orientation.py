import numpy as np
import cv2
from Fourier.utils.plotting import plot


def DFT(f):
    if len(f.shape) != 2:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    mask = np.indices(f.shape).sum(axis=0) % 2
    f = (-1) ** mask * f

    M, N = f.shape[:2]

    u, v = np.arange(M).reshape(-1, 1), np.arange(N).reshape(-1, 1)
    nv = v @ v.T
    mu = u @ u.T
    rows = np.exp(-2 * np.pi * 1j * nv / N)
    columns = np.exp(-2 * np.pi * 1j * mu / M)

    F = columns @ (f @ rows)
    F = np.log(F)

    return np.sqrt(F.real ** 2 + F.imag ** 2).astype(np.uint8)


def find_initial_orientation(img, r=50, th=0.95):
    f = DFT(img)
    xc, yc = f.shape[0] // 2, f.shape[1] // 2
    f = cv2.circle(f, (yc, xc), r, color=0, thickness=-1)
    th = th * np.max(f)
    f[f < th] = 0
    # plot(f)
    line = cv2.HoughLines(f, 1, np.pi / 180, 1)[0].flatten()
    theta = np.rad2deg(line[1])
    return 180 - theta


