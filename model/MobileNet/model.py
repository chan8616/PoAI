from pathlib import Path
from keras.utils.data_utils import get_file  # type: ignore
from keras.applications.mobilenet import MobileNet  # type: ignore

from model.model_config import ModelConfig
from model.keras_applications import model as modellib

ALPHA_DICT = {1.0: '1_0', 0.75: '7_5', 0.50: '5_0', 0.25: '2_5'}
ROWS_LIST = [128, 160, 192, 224]

ALPHA = 1.0
ROWS = ROWS_LIST[3]

ALPHA_TEXT = ALPHA_DICT.get(ALPHA)

TF_WEIGHTS_PATH_NO_TOP = (
        'https://github.com/fchollet/deep-learning-models/releases/download/'
        f'v0.6/mobilenet_{ALPHA_TEXT}_{ROWS}_tf_no_top.h5')


class Model(modellib.KerasAppBaseModel):
    def __init__(self, model_config=ModelConfig()):
        super(Model, self).__init__(MobileNet, model_config)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        weights_path = get_file(str(Path(TF_WEIGHTS_PATH_NO_TOP).name),
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
        return weights_path

    def set_trainable(self, layers):
        for layer in self.keras_model.layers[
                :layers]:
            layer.trainable = False
        for layer in self.keras_model.layers[
                layers:]:
            layer.trainable = True
