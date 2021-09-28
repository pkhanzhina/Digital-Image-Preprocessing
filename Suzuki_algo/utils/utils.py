import cv2
import numpy as np


def prepare_mask(img, hue_interval, saturation_interval, value_interval, blur=False):
    lower_bnd = [hue_interval[0], saturation_interval[0], value_interval[0]]
    upper_bnd = [hue_interval[1], saturation_interval[1], value_interval[1]]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = np.average((img_hsv <= upper_bnd) & (img_hsv >= lower_bnd), axis=-1) == 1
    if blur:
        mask = np.asarray(mask*255, dtype=np.uint8)
        mask = cv2.medianBlur(mask, 5) // 255
    return mask


def find_cnt_with_cv2(mask):
    mask = np.asarray(mask*255, dtype=np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return cnts
