from easydict import EasyDict

cfg = EasyDict()

cfg.hue_interval = (40, 88)
cfg.saturation_interval = (88, 163)
cfg.value_interval = (12, 140)

cfg.min_area = 10000
cfg.max_area = None

cfg.blur = True
