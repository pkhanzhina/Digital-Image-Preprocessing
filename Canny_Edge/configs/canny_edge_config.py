from easydict import EasyDict

cfg = EasyDict()

# Параметры фильтра Гаусса
cfg.smoothing_kernel_size = (5, 5)
cfg.smoothing_sigma = 1

# Параметры уточнения границ по порогам
cfg.low_threshold = 0.06
cfg.high_threshold = 0.18

