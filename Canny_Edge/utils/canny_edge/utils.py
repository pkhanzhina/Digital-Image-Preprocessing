import numpy as np


def direction_round(dir, nrof_dir):
    dir /= np.pi
    shift = 1 / nrof_dir
    angles = np.arange(-1, 1 + shift, 2 * shift)
    rounded_dir = np.zeros_like(dir)
    for i in range(nrof_dir):
        b, t = angles[i] - shift, angles[i] + shift
        angle = angles[i] if angles[i] != 1 else -1
        rounded_dir[(b <= dir) & (dir < t)] = angle
    return rounded_dir


def hysteresis(magnitude, low_threshold, up_threshold):
    maxx = np.max(magnitude)
    low_threshold, up_threshold = low_threshold * maxx, up_threshold * maxx
    hysteresis_arr = np.zeros_like(magnitude)
    hysteresis_arr[magnitude >= up_threshold] = 1
    hysteresis_arr[(low_threshold < magnitude) & (magnitude < up_threshold)] = 0.5
    return hysteresis_arr
