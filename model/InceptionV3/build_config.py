from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

InceptionV3LAYERS = LAYERS.copy()
InceptionV3LAYERS.update(OrderedDict([
    ('block_1', (0, 11)),
    ('block_2', (11, 18)),
    ('mixed_0', (18, 41)),
    ('mixed_1', (41, 64)),
    ('mixed_2', (64, 87)),
    ('mixed_3', (87, 101)),
    ('mixed_4', (101, 133)),
    ('mixed_5', (133, 165)),
    ('mixed_6', (165, 197)),
    ('mixed_7', (197, 229)),
    ('mixed_8', (229, 249)),
    ('mixed_9', (249, 280)),
    ('mixed_9', (281, 311)),
    ('heads', (311, None)),
    ]))


class InceptionV3Config(BuildConfig):
    NAME = 'InceptionV3'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (299, 299, 3)  # type: ignore
