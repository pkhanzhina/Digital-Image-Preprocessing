import cv2
import os
from algo.CannyEdge import CannyEdge
from configs.canny_edge_config import cfg as canny_cfg

data_path = 'data/'
canny_edge = CannyEdge(canny_cfg)

if __name__ == '__main__':
    img_name = 'emma.jpeg'

    img = cv2.imread(os.path.join(data_path, img_name))
    edges, nms, magn = canny_edge(img)

    cv2.imwrite('data/emma_madn.png', magn)
    cv2.imwrite('data/emma_edges.png', edges)
    cv2.imwrite('data/emma_nms.png', nms)


