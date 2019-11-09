from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

DenseNet169LAYERS = LAYERS.copy()
DenseNet169LAYERS.update(OrderedDict([
    ('block_0', (0, 7)),
    ('block_1', (7, 53)),
    ('block_2', (53, 141)),
    ('block_3', (141, 369)),
    ('block_4', (369, 594)),
    ('heads', (594, None)),
    ]))


class DenseNet169Config(BuildConfig):
    NAME = 'DenseNet169'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore
