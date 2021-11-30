import cv2
import numpy as np
from Canny_Edge.utils.canny_edge.utils import direction_round


def sobel_filtering(img, nrof_rounded_dir=None):
    kernel = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    dy = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    dx = cv2.filter2D(img, ddepth=-1, kernel=kernel.T)
    # dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    # dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    magn = np.sqrt(dx ** 2 + dy ** 2)
    dir = np.arctan2(dy, dx)
    if nrof_rounded_dir is not None:
        dir = direction_round(dir, nrof_dir=nrof_rounded_dir)
    return dir, magn