from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

DenseNet121LAYERS = LAYERS.copy()
DenseNet121LAYERS.update(OrderedDict([
    ('block_0', (0, 7)),
    ('block_1', (7, 53)),
    ('block_2', (53, 141)),
    ('block_3', (141, 313)),
    ('block_4', (313, 426)),
    ('heads', (426, None)),
    ]))


class DenseNet121Config(BuildConfig):
    NAME = 'DenseNet121'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore
