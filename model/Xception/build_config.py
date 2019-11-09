from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, LAYERS

XceptionLAYERS = LAYERS.copy()
XceptionLAYERS.update(OrderedDict([
    ('block_1', (0, 7)),
    ('block_2', (7, 16)),
    ('block_3', (16, 26)),
    ('block_4', (26, 36)),
    ('block_5', (36, 46)),
    ('block_6', (46, 56)),
    ('block_7', (56, 66)),
    ('block_8', (66, 76)),
    ('block_9', (76, 86)),
    ('block_10', (86, 96)),
    ('block_11', (96, 106)),
    ('block_12', (106, 116)),
    ('block_13', (116, 126)),
    ('block_14', (126, 132)),
    ('heads', (132, None)),
    ]))


class XceptionConfig(BuildConfig):
    NAME = 'Xception'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (299, 299, 3)  # type: ignore
