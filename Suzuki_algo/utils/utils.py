import cv2
import numpy as np


def prepare_mask(img, hue_interval, saturation_interval, value_interval, blur=False):
    lower_bnd = np.asarray([hue_interval[0], saturation_interval[0], value_interval[0]])
    upper_bnd = np.asarray([hue_interval[1], saturation_interval[1], value_interval[1]])
    if blur:
        img = cv2.medianBlur(img, 15)
        # img = cv2.GaussianBlur(img, (21, 21), 160)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_bnd, upper_bnd) // 255
    return mask


def find_cnt_with_cv2(mask):
    mask = np.asarray(mask*255, dtype=np.uint8)
    cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return cnts
