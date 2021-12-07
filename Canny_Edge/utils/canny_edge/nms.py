import numpy as np


def NMS(magnitude, directions, num=8):
    angles = np.arange(0, 1, 2 / num)
    directions = (directions + 1) % 1
    padded_magn = np.pad(magnitude, [(1, 1), (1, 1)], constant_values=[0, 0])
    h, w = directions.shape
    shifts = np.asarray([[[1, 0], [1, 2]],  # [[0, -1], [0, 1]] - (-180, 0)
                         [[0, 0], [2, 2]],  # [[-1, -1], [1, 1]] - (-135, 45)
                         [[0, 1], [2, 1]],  # [[-1, 0], [1, 0]] - (-90, 90)
                         [[0, 2], [2, 0]]])  # [[-1, 1], [1, -1]] - (-45, 135)
    nms = np.zeros_like(magnitude)
    for angle, cur_shift in zip(angles, shifts):
        ind = directions == angle
        crop_1 = padded_magn[cur_shift[0][0]:h + cur_shift[0][0], cur_shift[0][1]:w + cur_shift[0][1]]
        crop_2 = padded_magn[cur_shift[1][0]:h + cur_shift[1][0], cur_shift[1][1]:w + cur_shift[1][1]]
        nms[ind] = np.where((magnitude[ind] > crop_1[ind]) & (magnitude[ind] > crop_2[ind]), magnitude[ind], 0)
    return nms
