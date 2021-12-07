import numpy as np


def NMS(magnitude, directions, num=8):
    angles = np.arange(-1, 1, 2 / num)
    padded_magn = np.pad(magnitude, [(1, 1), (1, 1)], constant_values=[0, 0])
    h, w = directions.shape
    shifts = np.asarray([[[1, 0], [1, 2]],  # [[0, -1], [0, 1]] - (-180, 0)
                         [[0, 0], [2, 2]],  # [[-1, -1], [1, 1]] - (-135, 45)
                         [[0, 1], [2, 1]],  # [[-1, 0], [1, 0]] - (-90, 90)
                         [[0, 2], [2, 0]]])  # [[-1, 1], [1, -1]] - (-45, 135)
    nms = np.zeros_like(magnitude)
    for i in range(num):
        ind = directions == angles[i]
        cur_shift = shifts[i % 4]
        crop_1 = padded_magn[cur_shift[0][0]:h + cur_shift[0][0], cur_shift[0][1]:w + cur_shift[0][1]]
        crop_2 = padded_magn[cur_shift[1][0]:h + cur_shift[1][0], cur_shift[1][1]:w + cur_shift[1][1]]
        final_ind = np.zeros_like(ind)
        final_ind[ind] = (magnitude[ind] > crop_1[ind]) & (magnitude[ind] > crop_2[ind])
        nms[final_ind] = magnitude[final_ind]

    # _nms = np.zeros_like(magnitude)
    # for x in range(1, padded_magn.shape[0] - 1):
    #     for y in range(1, padded_magn.shape[1] - 1):
    #         if directions[x - 1, y - 1] == 0:
    #             if padded_magn[x, y] > padded_magn[x, y - 1] and padded_magn[x, y] > padded_magn[x, y + 1]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == 1 / 4:
    #             if padded_magn[x, y] > padded_magn[x - 1, y + 1] and padded_magn[x, y] > padded_magn[x + 1, y - 1]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == 1 / 2:
    #             if padded_magn[x, y] > padded_magn[x - 1, y] and padded_magn[x, y] > padded_magn[x + 1, y]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == 3 / 4:
    #             if padded_magn[x, y] > padded_magn[x - 1, y - 1] and padded_magn[x, y] > padded_magn[x + 1, y + 1]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == -1:
    #             if padded_magn[x, y] > padded_magn[x, y - 1] and padded_magn[x, y] > padded_magn[x, y + 1]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == - 3 / 4:
    #             if padded_magn[x, y] > padded_magn[x + 1, y - 1] and padded_magn[x, y] > padded_magn[x - 1, y + 1]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == - 1 / 2:
    #             if padded_magn[x, y] > padded_magn[x - 1, y] and padded_magn[x, y] > padded_magn[x + 1, y]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    #         if directions[x - 1, y - 1] == - 1 / 4:
    #             if padded_magn[x, y] > padded_magn[x + 1, y + 1] and padded_magn[x, y] > padded_magn[x - 1, y - 1]:
    #                 _nms[x - 1, y - 1] = padded_magn[x, y]
    # print(np.sum(_nms != nms))
    # print(np.unique(directions[_nms != nms]))
    return nms
