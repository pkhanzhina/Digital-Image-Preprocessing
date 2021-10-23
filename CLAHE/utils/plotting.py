import matplotlib.pyplot as plt
import cv2

def plot(img):
    plt.figure(figsize=(18, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
