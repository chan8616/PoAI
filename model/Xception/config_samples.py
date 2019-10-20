from ..keras_applications.build_config import BuildConfig, POOLINGS
from ..keras_applications.train_config import TrainConfig, LOSSES
#  from ..keras_applications.generator import TrainConfig, LOSSES
#  from ..keras_applications.config_samples import XceptionImagenetConfig
#  from generator.image_classification.image_classification_generator \
#          import DGC_CIFAR10


class ImagenetConfig(
        BuildConfig, TrainConfig):
    NAME = 'Xception'
    INPUT_SHAPE = (299, 299, 3)  # type: ignore

    POOLING = POOLINGS[1]

    CLASSES = 1000

    LOSS = LOSSES[2]


#  class CIFAR10(DGC_CIFAR10,
#                XceptionImagenetConfig):
#      def __init__(self):
#          super(CIFAR10, self).__init__()
#          self.TARGET_SIZE = self.INPUT_SHAPE[:2]
