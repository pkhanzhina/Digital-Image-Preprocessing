import os
from algo.CannyEdge import CannyEdge
from configs.canny_edge_config import cfg as canny_cfg
from utils.image_utils import save_image, load_image

data_path = 'data/'
canny_edge = CannyEdge(canny_cfg)

if __name__ == '__main__':
    img_name = 'emma.jpeg'

    img = load_image(os.path.join(data_path, img_name))
    edges = canny_edge(img)

    save_image(edges, f'data/{img_name}_edges.png')


