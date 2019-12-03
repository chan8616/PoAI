from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

MobileNetLAYERS = LAYERS.copy()
MobileNetLAYERS.update(OrderedDict([
    ('block_0', (0, 5)),
    ('block_1', (5, 12)),
    ('block_2', (12, 19)),
    ('block_3', (19, 26)),
    ('block_4', (26, 33)),
    ('block_5', (33, 40)),
    ('block_6', (40, 47)),
    ('block_7', (47, 54)),
    ('block_8', (54, 61)),
    ('block_9', (61, 68)),
    ('block_10', (68, 75)),
    ('block_11', (75, 82)),
    ('block_12', (82, 89)),
    ('block_13', (89, 96)),
    ('heads', (96, None)),
    ]))


class MobileNetConfig(BuildConfig):
    NAME = 'MobileNet'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore
