import cv2
from CLAHE.algoCLAHE import CLAHE
import os
from CLAHE.utils.plotting import plot


clahe = CLAHE(8, 8, 100)
data_path = 'data/'

if __name__ == '__main__':
    img_name = 'AHE.png'
    img = cv2.imread(os.path.join(data_path, img_name))
    contrast_img = clahe.apply(img)
    plot(contrast_img)
    cv2.imwrite(f"data/{img_name.split('.')[0]}_contrast.png", contrast_img)
