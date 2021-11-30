import cv2
import numpy as np

from Canny_Edge.algo.CCL import CCL
from Canny_Edge.utils.canny_edge.sobel_filtering import sobel_filtering
from Canny_Edge.utils.canny_edge.nms import NMS
from Canny_Edge.utils.canny_edge.utils import hysteresis


class CannyEdge:
    def __init__(self, cfg, nrof_dir=8):
        self.cfg = cfg
        self.nrof_dir = nrof_dir
        self.ccl = CCL(con_type=self.nrof_dir)

    def __call__(self, img):
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, self.cfg.smoothing_kernel_size, self.cfg.smoothing_sigma)
        img = np.asarray(img, dtype=np.float32)
        dir, magn = sobel_filtering(img, nrof_rounded_dir=self.nrof_dir)
        nms = NMS(magn, dir, self.nrof_dir)
        hyst = hysteresis(nms, self.cfg.low_threshold, self.cfg.high_threshold)
        components = self.ccl(hyst)
        print(np.sum(components), np.unique(components).shape)
        final_edge = np.zeros_like(img, dtype=np.uint8)
        for label in np.unique(components):
            if label == 0:
                continue
            ind = components == label
            if np.max(hyst[ind]) == 1:
                final_edge[ind] = 255
        return final_edge, nms, magn
