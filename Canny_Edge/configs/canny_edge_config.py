from easydict import EasyDict

cfg = EasyDict()

# Параметры фильтра Гаусса
cfg.smoothing_kernel_size = (3, 3)
cfg.smoothing_sigma = 21

# Параметры уточнения границ по порогам
cfg.low_threshold = 100
cfg.high_threshold = 200

