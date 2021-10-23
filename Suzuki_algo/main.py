import cv2
import os
from contour_detection.findContours import FindContours
from utils.utils import prepare_mask
from config.main_config import cfg
from utils.plotting import plot_cnt_as_bbox, plot_cnt, plot

cnt_detector = FindContours()

data_path = 'data/'

if __name__ == '__main__':
    img_name = 'segment.jpg'

    img_bgr = cv2.imread(os.path.join(data_path, img_name))

    img_mask = prepare_mask(img_bgr, cfg.hue_interval, cfg.saturation_interval, cfg.value_interval,
                            blur=cfg.blur)

    plot(img_mask*255)

    cnts, _ = cnt_detector(img_mask)

    plot_cnt_as_bbox(img_bgr.copy(), cnts,
                     draw_contours=True,
                     min_area=cfg.min_area, max_area=cfg.max_area,
                     path_to_save=f"data/{img_name.split('.')[0]}_bbox.jpg")

    plot_cnt(img_bgr.copy(), cnts,
             path_to_save=f"data/{img_name.split('.')[0]}_cnt.jpg")
