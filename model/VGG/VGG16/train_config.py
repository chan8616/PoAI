from argparse import Namespace
from gooey import GooeyParser

from model.keras_applications.train_config import (
        WEIGHTS, LOSSES, OPTIMIZERS, TrainConfig)
from .build_config import VGG16LAYERS


class VGG16TrainConfig(TrainConfig):
    #  WEIGHT = None

    #  EPOCHS = 10
    #  VALIDATION_STEPS = 10

    LOSS = LOSSES[2]

    #  OPTIMIZER = OPTIMIZERS[0]
    #  LEARNING_RATE = 1e-2
    #  LEARNING_MOMENTUM = 0.9

    #  MONITOR = 'loss'

    def __init__(self):
        super(VGG16TrainConfig, self).__init__()
        for k, v in VGG16LAYERS.items():
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
