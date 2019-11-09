from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

DenseNet201LAYERS = LAYERS.copy()
DenseNet201LAYERS.update(OrderedDict([
    ('block_0', (0, 7)),
    ('block_1', (7, 53)),
    ('block_2', (53, 141)),
    ('block_3', (141, 481)),
    ('block_4', (481, 706)),
    ('heads', (706, None)),
    ]))


class DenseNet201Config(BuildConfig):
    NAME = 'DenseNet201'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore
