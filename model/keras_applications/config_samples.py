from .build_config import BuildConfig, POOLINGS
from .train_config import TrainConfig, LOSSES


class XceptionImagenetConfig(
        BuildConfig, TrainConfig):
    NAME = 'Xception'
    INPUT_SHAPE = (299, 299, 3)  # type: ignore

    POOLING = POOLINGS[1]

    CLASSES = 1000

    LOSS = LOSSES[2]
    LEARNING_RATE = 0.045
