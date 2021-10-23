import numpy as np


def count_notna_cdfs(cdfs):
    c = 0
    for cdf in cdfs:
        if cdf is None:
            c += 1
    return c


def processing_zone(img, control_points, cdfs):
    new_img = np.zeros_like(img)
    x1, y1 = 0, 0
    x2, y2 = control_points
    cdf_a, cdf_b, cdf_c, cdf_d = cdfs

    zone_type = count_notna_cdfs(cdfs)

    for i in range(x1, x2):
        for j in range(y1, y2):
            if zone_type == 3:
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
            elif zone_type == 2:
                if cdf_a is not None and cdf_b is not None:
                    n = (y2 - j) / (y2 - y1)
                    cdf = n * cdf_a + (1 - n) * cdf_b
                elif cdf_b is not None and cdf_c is not None:
                    m = (x2 - i) / (x2 - x1)
                    cdf = m * cdf_b + (1 - m) * cdf_c
                elif cdf_c is not None and cdf_d is not None:
                    n = (y2 - j) / (y2 - y1)
                    cdf = n * cdf_d + (1 - n) * cdf_c
                elif cdf_a is not None and cdf_d is not None:
                    m = (x2 - i) / (x2 - x1)
                    cdf = m * cdf_a + (1 - m) * cdf_d
                else:
                    continue
                curr_value = img[i, j]
                new_img[i, j] = cdf[curr_value]
            else:
                m = (x2 - i) / (x2 - x1)
                n = (y2 - j) / (y2 - y1)
                cdf = m * (n * cdf_a + (1 - n) * cdf_b) + (1 - m) * (n * cdf_d + (1 - n) * cdf_c)
                curr_value = img[i, j]
                new_img[i, j] = cdf[curr_value]
    return new_img
