import matplotlib.pyplot as plt


def plot_img(img, cmap=None, title=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
