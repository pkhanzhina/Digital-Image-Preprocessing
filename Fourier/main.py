import cv2
from Fourier.utils.plotting import plot, rotate_image
from Fourier.find_init_orientation import find_initial_orientation


if __name__ == '__main__':
    img = cv2.imread('data/2021-12-16 21.30.08.jpg')

    img = rotate_image(img, -90)
    plot(img)
    theta = find_initial_orientation(img, r=50, th=0.95)

    rot_img = rotate_image(img, theta)
    plot(rot_img)


