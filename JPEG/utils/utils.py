import cv2
import numpy as np


def sobel_filtering(img, theta=np.pi / 18, min_E=30):
    dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    E = np.sqrt(dx ** 2 + dy ** 2)
    dx[E <= min_E] = 0
    dy[E <= min_E] = 0
    dir = np.arctan2(dy, dx)
    mask = np.ones_like(dir)
    mask[(0 <= dir) & (dir <= theta) |
         ((np.pi / 2 - theta <= dir) & (dir <= np.pi / 2 + theta)) |
         ((np.pi - theta <= dir) & (dir < np.pi))] = 0
    return mask


def get_df(img):
    kernel = np.array([[-1, 2, -1]])
    dx = cv2.filter2D(img, -1, kernel)
    dy = cv2.filter2D(img, -1, kernel.T)
    return np.abs(dx), np.abs(dy)
