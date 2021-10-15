import numpy as np
import cv2


class AHE:
    def __init__(self, N):
        self.N = N

    def __call__(self, img):
        img_map = self.prepare_color_map(img)

        h, w = img.shape[:2]
        grid_h, grid_w = np.linspace(0, h - 1, self.N, dtype=int), np.linspace(0, w - 1, self.N, dtype=int)
        control_coords_x = np.hstack(([grid_h[0]], (grid_h[:-1] + grid_h[1:]) // 2, [grid_h[-1]]))
        control_coords_y = np.hstack(([grid_w[0]], (grid_w[:-1] + grid_w[1:]) // 2, [grid_w[-1]]))

        cdfs = {}
        for i in range(0, len(control_coords_x)):
            for j in range(0, len(control_coords_y)):
                if i - 1 >= 0 and i < self.N and j - 1 >= 0 and j < self.N:
                    xt, yt = grid_h[i - 1], grid_w[j - 1]
                    xb, yb = grid_h[i], grid_w[j]
                    counts, bins = np.histogram(img[xt:xb, yt:yb].flatten(), bins=np.arange(0, 257))
                    counts = counts / np.sum(counts)
                    cdf = np.cumsum(counts) * 255
                else:
                    cdf = None
                cdfs[i, j] = cdf

        new_img = np.zeros_like(img)
        for i in range(0, len(control_coords_x) - 1):
            for j in range(0, len(control_coords_y) - 1):
                x1, x3 = control_coords_x[i], control_coords_x[i + 1]
                y1, y3 = control_coords_y[j], control_coords_y[j + 1]
                cdf1, cdf2, cdf3, cdf4 = cdfs[i, j], cdfs[i, j + 1], cdfs[i + 1, j + 1], cdfs[i + 1, j]
                new_img[x1:x3, y1:y3] = self.zone(img=img[x1:x3, y1:y3],
                                                  img_map=img_map[x1:x3, y1:y3],
                                                  control_points=[[0, 0, x3 - x1, y3 - y1],
                                                                  [cdf1, cdf2, cdf3, cdf4]])
        return new_img

    def zone(self, img, img_map, control_points):
        h, w = img.shape[:2]
        new_img = np.zeros_like(img)
        x1, y1, x2, y2 = control_points[0]
        cdf_a = control_points[-1][0]
        cdf_b = control_points[-1][1]
        cdf_c = control_points[-1][2]
        cdf_d = control_points[-1][3]

        for i in range(0, h):
            for j in range(0, w):
                if img_map[i, j] == 0:
                    if cdf_a is not None:
                        cdf = cdf_a
                    elif cdf_b is not None:
                        cdf = cdf_b
                    elif cdf_c is not None:
                        cdf = cdf_c
                    else:
                        cdf = cdf_d
                    curr_value = img[i, j]
                    new_img[i, j] = cdf[curr_value]
                elif img_map[i, j] == 1:
                    if cdf_a is not None and cdf_b is not None:
                        n = 0 if y2 - y1 == 0 else (y2 - j) / (y2 - y1)
                        cdf = n * cdf_a + (1 - n) * cdf_b
                    elif cdf_b is not None and cdf_c is not None:
                        m = 0 if x2 - x1 == 0 else (x2 - i) / (x2 - x1)
                        cdf = m * cdf_b + (1 - m) * cdf_c
                    elif cdf_c is not None and cdf_d is not None:
                        n = 0 if y2 - y1 == 0 else (y2 - j) / (y2 - y1)
                        cdf = n * cdf_d + (1 - n) * cdf_c
                    elif cdf_a is not None and cdf_d is not None:
                        m = 0 if x2 - x1 == 0 else (x2 - i) / (x2 - x1)
                        cdf = m * cdf_a + (1 - m) * cdf_d
                    else:
                        continue
                    curr_value = img[i, j]
                    new_img[i, j] = cdf[curr_value]
                elif img_map[i, j] == 2:
                    m = 0 if x2 - x1 == 0 else (x2 - i) / (x2 - x1)
                    n = 0 if y2 - y1 == 0 else (y2 - j) / (y2 - y1)
                    cdf = m * (n * cdf_a + (1 - n) * cdf_b) + (1 - m) * (n * cdf_d + (1 - n) * cdf_c)
                    curr_value = img[i, j]
                    new_img[i, j] = cdf[curr_value]
                elif img_map[i, j] == 3:
                    new_img[i, j] = img[i, j]
        return new_img

    def prepare_color_map(self, img):
        segm_map = np.ones_like(img)
        h, w = img.shape[:2]
        grid_h = np.linspace(0, h - 1, self.N, dtype=int)
        grid_w = np.linspace(0, w - 1, self.N, dtype=int)
        control_points = []
        for i in [1, self.N - 1]:
            for j in [1, self.N - 1]:
                xt, yt = grid_h[i - 1], grid_w[j - 1]
                xb, yb = grid_h[i], grid_w[j]
                xc, yc = (xb + xt) // 2, (yb + yt) // 2

                hh = 0 if i == 1 else h
                ww = 0 if j == 1 else w

                segm_map[min(hh, xc):max(hh, xc), min(ww, yc):max(ww, yc)] = 0
                control_points.append((xc, yc))

        c00, c11 = control_points[0], control_points[-1]
        segm_map[c00[0]:c11[0], c00[1]:c11[1]] = 2

        for i in range(1, grid_h.shape[0]):
            for j in range(1, grid_w.shape[0]):
                x, y = grid_h[i], grid_w[j]
                segm_map[x, y] = 3
        return segm_map

ahe = AHE(8)

if __name__ == '__main__':
    img_dir = 'data/test.png'
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast_img = ahe(img)
    cv2.imwrite('data/test_contrast.png', contrast_img)
