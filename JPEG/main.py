import cv2
import os
import numpy as np
from utils.utils import sobel_filtering, get_df
from utils.BAG import BAG

data_path = 'data'

if __name__ == '__main__':
    img_name = 'planes_forg_2.jpg'
    img = cv2.cvtColor(cv2.imread(os.path.join(data_path, img_name)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = sobel_filtering(img)
    anomaly_score = BAG(img, mask=mask, AC=33)
    cv2.imwrite(f"{data_path}/{img_name.split('.')[0]}_anomaly_score.png", anomaly_score)
