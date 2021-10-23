import cv2
import numpy as np
from Suzuki_algo.contour_detection.utils import NeighboringPoints


class FindContours:
    def __init__(self):
        self.neighbors = NeighboringPoints()

    def __call__(self, mask):
        """
        :return:
            cnt - список контуров
            hierarchy - список родителя каждого контура: hierarchy[i] - номер родителя в списке cnt
                ("-1" - в качестве родителя берется frame)
        """
        h, w = mask.shape[:2]
        f = np.asarray(mask.copy(), dtype=np.int)
        NBD = 1
        cnts, hierarchy = [], []
        is_hole = []
        for i in range(h):
            LNBD = 1
            for j in range(w):
                if f[i, j] == 0:
                    continue
                if j > 0 and f[i, j] == 1 and f[i, j - 1] == 0:
                    NBD += 1
                    is_hole.append(False)
                    if LNBD == 1:
                        hierarchy.append((-1, NBD))
                    else:
                        hierarchy.append((LNBD - 2, NBD) if is_hole[LNBD - 2] else (hierarchy[LNBD - 2][0], NBD))
                    cnts.append(self._border_detection(f, np.asarray([i, j]), np.asarray([0, -1]), NBD)[:, ::-1])
                elif j < w - 1 and f[i, j] >= 1 and f[i, j + 1] == 0:
                    NBD += 1
                    if f[i, j] > 1:
                        LNBD = f[i, j]
                    is_hole.append(True)
                    if LNBD == 1:
                        hierarchy.append((-1, NBD))
                    else:
                        hierarchy.append((hierarchy[LNBD - 2][0], NBD) if is_hole[LNBD - 2] else (LNBD - 2, NBD))
                    cnts.append(self._border_detection(f, np.asarray([i, j]), np.asarray([0, 1]), NBD)[:, ::-1])
                if f[i, j] != 1:
                    LNBD = abs(f[i, j])
        return cnts, hierarchy

    def _border_detection(self, f, p, delta, NBD):
        w = f.shape[1]
        p2 = p + delta
        cnt = [p]
        p1 = self._get_nonzero_pixel(p, p2, f, clockwise=True)
        if p1 is None:
            f[p[0], p[1]] = -NBD
            return np.array(cnt)
        p2, p3 = p1, p
        while True:
            p4 = self._get_nonzero_pixel(p3, p2, f, clockwise=False)
            if p3[1] < w - 1:
                if f[p3[0], p3[1] + 1] == 0:
                    f[p3[0], p3[1]] = -NBD
                elif f[p3[0], p3[1]] == 1:
                    f[p3[0], p3[1]] = NBD
                cnt.append(p3)
            if p4 is None:
                break
            if p4[0] == p[0] and p4[1] == p[1] and p3[0] == p1[0] and p3[1] == p1[1]:
                break
            p2, p3 = p3, p4
        return np.array(cnt)

    def _get_nonzero_pixel(self, start_point, work_point, f, clockwise: bool):
        h, w = f.shape[:2]
        nonzero_point = None
        for delta in self.neighbors.get(work_point - start_point, clockwise=clockwise):
            next_p = start_point + delta
            if next_p[0] > h - 1 or next_p[1] > w - 1 or next_p[0] < 0 or next_p[1] < 0:
                continue
            if f[next_p[0], next_p[1]] != 0:
                nonzero_point = next_p
                break
        return nonzero_point
