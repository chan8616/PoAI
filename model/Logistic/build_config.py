from collections import OrderedDict

from model.keras_applications.build_config import (
        BuildConfig, POOLINGS)


class LogisticConfig(BuildConfig):
    NAME = 'Logistic'

    INPUT_SHAPE = []  # type: ignore

    POOLING = POOLINGS[0]
    HIDDEN_LAYERS = []
    CLASSES = 0
