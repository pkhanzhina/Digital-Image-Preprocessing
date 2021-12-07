import cv2
import os
import numpy as np
from algo.CannyEdge import CannyEdge
from configs.canny_edge_config import cfg as canny_cfg

data_path = 'data/'
canny_edge = CannyEdge(canny_cfg)

if __name__ == '__main__':
    img_name = 'emma.jpeg'

    img = cv2.imread(os.path.join(data_path, img_name))
    edges = canny_edge(img)

    cv2.imwrite(f'data/{img_name}_edges.png', edges)


