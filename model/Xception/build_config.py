from collections import OrderedDict

from model.keras_applications.build_config import BuildConfig, POOLINGS, LAYERS

XceptionLAYERS = LAYERS.copy()
XceptionLAYERS.update(OrderedDict([
      ('block_1', (0, 7)),
      ('block_2', (7, 16)),
      *[('block_{}'.format(i-1), (10*i-4, 10*i+6))
          for i in range(3, 13)],
      ('13', (126, 132)),
      ('heads', (132, None)),
      ('entry_flow', (0, 36)),
      ('middle_flow', (36, 116)),
      ('exit_flow', (116, None)),
      ]))


class XceptionConfig(BuildConfig):
    NAME = 'Xception'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (299, 299, 3)  # type: ignore

    POOLING = POOLINGS[1]
