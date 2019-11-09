from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

VGG19LAYERS = LAYERS.copy()
VGG19LAYERS.update(OrderedDict([
    ('block_1', (0, 4)),
    ('block_2', (4, 7)),
    ('block_3', (7, 12)),
    ('block_4', (12, 17)),
    ('block_5', (17, 22)),
    ('heads', (22, None)),
    ]))


class VGG19Config(BuildConfig):
    NAME = 'VGG19'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore
