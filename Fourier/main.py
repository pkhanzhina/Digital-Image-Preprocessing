import cv2
from Fourier.utils.plotting import plot, rotate_image
from Fourier.find_init_orientation import find_initial_orientation


if __name__ == '__main__':
    img = cv2.imread('data/temp.jpg')

    img = rotate_image(img, 20)
    plot(img)
    theta = find_initial_orientation(img, r=5, th=0.80)
    print(f"angle: {theta}")
    rot_img = rotate_image(img, -theta)
    plot(rot_img)


