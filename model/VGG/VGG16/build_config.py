from collections import OrderedDict

from model.keras_applications.build_config import (
        BuildConfig, LAYERS, POOLINGS)

VGG16LAYERS = LAYERS.copy()
VGG16LAYERS.update(OrderedDict([
    ('block_1', (0, 4)),
    ('block_2', (4, 7)),
    ('block_3', (7, 11)),
    ('block_4', (11, 15)),
    ('block_5', (15, 19)),
    ('heads', (19, None)),
    ]))


class VGG16Config(BuildConfig):
    NAME = 'VGG16'

    INPUT_SHAPE = (224, 224, 3)  # type: ignore

    POOLING = POOLINGS[0]
    HIDDEN_LAYERS = [4096, 4096]
    CLASSES = 1000
