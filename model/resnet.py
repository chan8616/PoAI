#
# -*- coding: utf-8 -*-

import numpy as np
import copy

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Lambda,\
#                                             AveragePooling2D, ZeroPadding2D, Flatten, \
#                                             Activation, add, BatchNormalization, Layer, InputSpec
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras import initializers
# from tensorflow.python.keras import backend as K
#
# import sys
# sys.setrecursionlimit(3000)
#
from .model import NET
#
# class Scale(Layer):
#     '''Custom Layer for ResNet used for BatchNormalization.
#
#     Learns a set of weights and biases used for scaling the input data.
#     the output consists simply in an element-wise multiplication of the input
#     and a sum of a set of constants:
#         out = in * gamma + beta,
#     where 'gamma' and 'beta' are the weights and biases larned.
#     # Arguments
#         axis: integer, axis along which to normalize in mode 0. For instance,
#             if your input tensor has shape (samples, channels, rows, cols),
#             set axis to 1 to normalize per feature map (channels axis).
#         momentum: momentum in the computation of the
#             exponential average of the mean and standard deviation
#             of the data, for feature-wise normalization.
#         weights: Initialization weights.
#             List of 2 Numpy arrays, with shapes:
#             `[(input_shape,), (input_shape,)]`
#         beta_init: name of initialization function for shift parameter
#             (see [initializers](../initializers.md)), or alternatively,
#             Theano/TensorFlow function to use for weights initialization.
#             This parameter is only relevant if you don't pass a `weights` argument.
#         gamma_init: name of initialization function for scale parameter (see
#             [initializers](../initializers.md)), or alternatively,
#             Theano/TensorFlow function to use for weights initialization.
#             This parameter is only relevant if you don't pass a `weights` argument.
#     '''
#     def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
#         self.momentum = momentum
#         self.axis = axis
#         self.beta_init = initializers.get(beta_init)
#         self.gamma_init = initializers.get(gamma_init)
#         self.initial_weights = weights
#         super(Scale, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.input_spec = [InputSpec(shape=input_shape)]
#         shape = (int(input_shape[self.axis]),)
#
#         self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
#         self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
#         self.trainable_weights.append(self.gamma)
#         self.trainable_weights.append(self.beta)
#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#
#     def call(self, x, mask=None):
#         input_shape = self.input_spec[0].shape
#         broadcast_shape = [1] * len(input_shape)
#         broadcast_shape[self.axis] = input_shape[self.axis]
#
#         out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
#         return out
#
#     def get_config(self):
#         config = {"momentum": self.momentum, "axis": self.axis}
#         base_config = super(Scale, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     '''The identity_block is the block that has no conv layer at shortcut
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the nb_filters of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     '''
#     eps = 1.1e-5
#     nb_filter1, nb_filter2, nb_filter3 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#     scale_name_base = 'scale' + str(stage) + block + '_branch'
#
#     x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
#     x = Activation('relu', name=conv_name_base + '2a_relu')(x)
#
#     x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
#     x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
#     x = Activation('relu', name=conv_name_base + '2b_relu')(x)
#
#     x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
#     x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)
#
#     x = add([x, input_tensor], name='res' + str(stage) + block)
#     x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
#     return x
#
# def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#     '''conv_block is the block that has a conv layer at shortcut
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the nb_filters of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
#     And the shortcut should have subsample=(2,2) as well
#     '''
#     eps = 1.1e-5
#     nb_filter1, nb_filter2, nb_filter3 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#     scale_name_base = 'scale' + str(stage) + block + '_branch'
#
#     x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
#     x = Activation('relu', name=conv_name_base + '2a_relu')(x)
#
#     x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
#     x = Conv2D(nb_filter2, (kernel_size, kernel_size),
#                       name=conv_name_base + '2b', use_bias=False)(x)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
#     x = Activation('relu', name=conv_name_base + '2b_relu')(x)
#
#     x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
#     x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)
#
#     shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
#                              name=conv_name_base + '1', use_bias=False)(input_tensor)
#     shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
#     shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)
#
#     x = add([x, shortcut], name='res' + str(stage) + block)
#     x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
#     return x
#
# def resnet152_model(shape,weights):
#     '''Instantiate the ResNet152 architecture,
#     # Arguments
#         weights_path: path to pretrained weight file
#     # Returns
#         A Keras model instance.
#     '''
#     eps = 1.1e-5
#
#     # Handle Dimension Ordering for different backends
#     global bn_axis
#     bn_axis = 3
#     img_input = Input(shape=shape, name='image')
#     img_resize = Lambda(lambda img: tf.image.resize_images(img, (224, 224)))(img_input)
#     # img_input = keras.backend.resize_images(img_input, 49, 49, "channels_last")
#     x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_resize)
#     x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
#     x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
#     x = Scale(axis=bn_axis, name='scale_conv1')(x)
#     x = Activation('relu', name='conv1_relu')(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
#
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#     for i in range(1,8):
#         x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))
#
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     for i in range(1,36):
#         x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))
#
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#
#     x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
#     x_fc = Flatten()(x_fc)
#     x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
#
#     model = Model(img_input, x_fc)
#
#     # load weights
#     if weights:
#         from google_drive_downloader import GoogleDriveDownloader as gdd
#         weight_path = '/tmp/resnet152_weights_tf.h5'
#         gdd.download_file_from_google_drive(file_id='0Byy2AcGyEVxfeXExMzNNOHpEODg',
#                                             dest_path=weight_path)
#         # try:
#         model.load_weights(weight_path, by_name=True)
#         # except:
#             # print("[!] fail to download resnet 152 weights")
#     return model
# -*- coding: utf-8 -*-
"""ResNet152 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adaptation of code from flyyufelix, mvoelk, BigMoyan, fchollet
"""

import numpy as np
import warnings

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Layer, InputSpec, Lambda
from tensorflow.python.keras.utils import get_file

import sys
sys.setrecursionlimit(3000)

WEIGHTS_PATH = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5'

class Scale(Layer):
    """Custom Layer for ResNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    Keyword arguments:
    axis -- integer, axis along which to normalize in mode 0. For instance,
        if your input tensor has shape (samples, channels, rows, cols),
        set axis to 1 to normalize per feature map (channels axis).
    momentum -- momentum in the computation of the exponential average
        of the mean and standard deviation of the data, for
        feature-wise normalization.
    weights -- Initialization weights.
        List of 2 Numpy arrays, with shapes:
        `[(input_shape,), (input_shape,)]`
    beta_init -- name of initialization function for shift parameter
        (see [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.
    gamma_init -- name of initialization function for scale parameter (see
        [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.

    """
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights.append(self.gamma)
        self.trainable_weights.append(self.beta)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity_block is the block that has no conv layer at shortcut

    Keyword arguments
    input_tensor -- input tensor
    kernel_size -- defualt 3, the kernel size of middle conv layer at main path
    filters -- list of integers, the nb_filters of 3 conv layer at main path
    stage -- integer, current stage label, used for generating layer names
    block -- 'a','b'..., current block label, used for generating layer names

    """
    eps = 1.1e-5

    bn_axis = 3

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    Keyword arguments:
    input_tensor -- input tensor
    kernel_size -- defualt 3, the kernel size of middle conv layer at main path
    filters -- list of integers, the nb_filters of 3 conv layer at main path
    stage -- integer, current stage label, used for generating layer names
    block -- 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well

    """
    eps = 1.1e-5

    bn_axis = 3

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def ResNet152(include_top=True, weights=None,
              input_tensor=None, input_shape=None,
              large_input=False, pooling=None,
              classes=1000):
    """Instantiate the ResNet152 architecture.

    Keyword arguments:
    include_top -- whether to include the fully-connected layer at the
        top of the network. (default True)
    weights -- one of `None` (random initialization) or "imagenet"
        (pre-training on ImageNet). (default None)
    input_tensor -- optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.(default None)
    input_shape -- optional shape tuple, only to be specified if
        `include_top` is False (otherwise the input shape has to be
        `(224, 224, 3)` (with `channels_last` data format) or
        `(3, 224, 224)` (with `channels_first` data format). It should
        have exactly 3 inputs channels, and width and height should be
        no smaller than 197. E.g. `(200, 200, 3)` would be one valid value.
        (default None)
    large_input -- if True, then the input shape expected will be
        `(448, 448, 3)` (with `channels_last` data format) or
        `(3, 448, 448)` (with `channels_first` data format). (default False)
    pooling -- Optional pooling mode for feature extraction when
        `include_top` is `False`.
        - `None` means that the output of the model will be the 4D
            tensor output of the last convolutional layer.
        - `avg` means that global average pooling will be applied to
            the output of the last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
        (default None)
    classes -- optional number of classes to classify image into, only
        to be specified if `include_top` is True, and if no `weights`
        argument is specified. (default 1000)

    Returns:
    A Keras model instance.

    Raises:
    ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    eps = 1.1e-5

    if large_input:
        img_size = 448
    else:
        img_size = 224

    # Determine proper input shape
    img_input = input_tensor

    # handle dimension ordering for different backends
    bn_axis = 3

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if large_input:
        x = AveragePooling2D((14, 14), name='avg_pool')(x)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # include classification layer by default, not included for feature extraction
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet152')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet152_weights_tf.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='cdb18a2158b88e392c0905d47dcef965')
        else:
            weights_path = get_file('resnet152_weights_tf_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='4a90dcdafacbd17d772af1fb44fc2660')
        model.load_weights(weights_path, by_name=True)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model

class RESNET152(NET):

    def __init__(self, **kargs):
        kargs['model'] = 'resnet152'
        kargs['init'] = ['imagenet', None]
        super(RESNET152, self).__init__(**kargs)

    def build_model(self, conf):

        img = img_input = Input(shape=conf['input_shape'], name='image')
        img_resize = Lambda(lambda img: tf.image.resize_images(img, (224, 224)))(img)
        base_model = ResNet152(input_tensor=img_resize, weights=conf['init'], include_top=False)
        x = base_model.output
        x = Flatten()(x)
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=img, outputs=y_pred)

        if conf['freeze'] and conf['init']:
            for layer in base_model.layers:
                layer.trainable = False
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                            metrics=['accuracy'])
