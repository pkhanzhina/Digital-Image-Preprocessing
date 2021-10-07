import cv2
import matplotlib.pyplot as plt


def plot_cnt_as_bbox(img, cnts, draw_contours=False, min_area=None, max_area=None, path_to_save=None):
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if min_area is not None and area < min_area or \
                max_area is not None and area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        if draw_contours:
            img = cv2.drawContours(img, [cnt], -1, color=(0, 0, 255), thickness=2)
    if path_to_save is not None:
        cv2.imwrite(path_to_save, img)

    plt.figure(figsize=(18, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def plot_cnt(img, cnts, path_to_save=None):
    img_cnt = cv2.drawContours(img, cnts, -1, color=(0, 0, 255), thickness=2)
    if path_to_save is not None:
        cv2.imwrite(path_to_save, img_cnt)

    plt.figure(figsize=(18, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def plot(img):
    plt.figure(figsize=(18, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
