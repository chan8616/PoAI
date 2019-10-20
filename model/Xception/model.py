from pathlib import Path
from keras.utils.data_utils import get_file  # type: ignore
from keras.applications import Xception  # type: ignore

from ..model_config import ModelConfig
from ..keras_applications import model as modellib

TF_WEIGHTS_PATH_NO_TOP = \
        'https://github.com/fchollet/' \
        'deep-learning-models/releases/download/v0.4/' \
        'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


class XceptionModel(modellib.KerasAppBaseModel):
    def __init__(self, model_config=ModelConfig()):
        super(XceptionModel, self).__init__(Xception, model_config)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        weights_path = get_file(str(Path(TF_WEIGHTS_PATH_NO_TOP).name),
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                file_hash='b0042744bf5b25fce3cb969f33bebb97')
        return weights_path
