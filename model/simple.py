import tensorflow as tf
import numpy as np

from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, \
                                            AveragePooling2D, ZeroPadding2D, Flatten, \
                                            Activation, add, BatchNormalization, Layer, InputSpec
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K

import os
import sys
sys.path.append(os.path.abspath('..')) # refer to sibling folder

from .ops import *
from utils.data import *
from utils.util import *
from .model import NET

class LOGISTIC(NET):
    """

    Scenario 1 : Using open data.
     In that case, following arguments are useless
        : classes, image_size, train_data, test_data, train_label, test_label

    Scenario 2 : Hand-crafted dataset.
     In that case, following arguments are useless.
        : dataset

    """
    def __init__(self, **kargs):
        kargs['model'] = 'logistic'
        kargs['init'] = [None, None]

        super(LOGISTIC, self).__init__(**kargs)
    def build_model(self, conf):

        input_ = Input(shape=(conf['input_shape']), name='data')
        x = Flatten()(input_)
#        x = Dense(512, activation='relu', name='lin')(x)
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)

        self.model = Model(inputs=input_, outputs=y_pred)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                            metrics=['accuracy'])
