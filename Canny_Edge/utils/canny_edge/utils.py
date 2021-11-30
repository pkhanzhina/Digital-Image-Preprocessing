import numpy as np


def direction_round(dir, nrof_dir):
    intervals = np.linspace(-np.pi / nrof_dir, 2 * np.pi - 1 / nrof_dir, nrof_dir + 1)
    angles = np.arange(0, 2, 2 / nrof_dir)
    rounded_dir = np.zeros_like(dir)
    for i in range(nrof_dir):
        b, t = intervals[i], intervals[i + 1]
        rounded_dir[(b <= dir) & (dir < t)] = angles[i]
    return rounded_dir


def hysteresis(magnitude, low_threshold, up_threshold):
    hysteresis_arr = np.zeros_like(magnitude)
    hysteresis_arr[magnitude >= up_threshold] = 1
    hysteresis_arr[(low_threshold < magnitude) & (magnitude< up_threshold)] = 0.5
    return hysteresis_arr
