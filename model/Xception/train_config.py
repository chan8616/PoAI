from collections import OrderedDict

from model.keras_applications.train_config import (
        WEIGHTS, LOSSES, OPTIMIZERS, TrainConfig)
from .build_config import XceptionLAYERS


class XceptionTrainConfig(TrainConfig):
    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(XceptionTrainConfig, self).__init__()
        for k, v in XceptionLAYERS.items():
            if 'all' == k:
                pass
            elif 'heads' == k:
                pass
            elif v[0] == 0:
                self.TRAIN_LAYERS.update([
                    ('{}+ (all)'.format(k), v[0])
                    ])
            else:
                self.TRAIN_LAYERS.update([
                    ('{}+'.format(k), v[0])
                    ])
