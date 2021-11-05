import cv2
import os
from binarization.utils.Sauvola_algo import sauvola_binarization
from binarization.utils.Otsu_algo import otsu_binarization

data_path = 'data/'

if __name__ == '__main__':
    # img_name = 'test.png'
    img_name = 'lena.jpg'

    img = cv2.imread(os.path.join(data_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sauvola_bin_img = sauvola_binarization(img)
    cv2.imwrite(f"data/{img_name.split('.')[0]}_sauvola.png", sauvola_bin_img)

    otsu_bin_img = otsu_binarization(img)
    cv2.imwrite(f"data/{img_name.split('.')[0]}_otsu.png", otsu_bin_img)


