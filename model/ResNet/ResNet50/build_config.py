from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, POOLINGS, LAYERS

ResNet50LAYERS = LAYERS.copy()
ResNet50LAYERS.update(OrderedDict([
    ('stage_1', (0, 6)),
    ('stage_2', (6, 38)),
    ('stage_3', (38, 80)),
    ('stage_4', (80, 142)),
    ('stage_5', (142, 174)),
    ('heads', (174, None)),
    ]))


class ResNet50Config(BuildConfig):
    NAME = 'ResNet50'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore
