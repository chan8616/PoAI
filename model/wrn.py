import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath('..')) # refer to sibling folder

from .ops import *
from utils.data import *
from utils.util import *
from .model import NET

class WRN(NET):
    """

    Scenario 1 : Using open data.
     In that case, following arguments are useless
        : classes, image_size, train_data, test_data, train_label, test_label

    Scenario 2 : Hand-crafted dataset.
     In that case, following arguments are useless.
        : dataset

    """
    def __init__(self,
                 sess,               # tf.Session()
                 dataset,     # dataset name, open data e.g. mnist or hand-crafted dataset
                 classes,     #
                 image_size,  # if it is given, image can be cropped or scaling
                 model_name = 'simple_logistic',
                 learning_rate = 1e-4,
                 optimizer = 'gradient',
                 beta1 = 0.5,   # (optional) for some optimizer
                 beta2 = 0.99,  # (optional) for some optimizer
                 batch_size = 64,
                 epochs = 20,
                 model_dir = None
                 ):

        super(WRN, self).__init__(sess,
                     model_name,
                     dataset,
                     learning_rate,
                     optimizer,
                     beta1,
                     beta2,
                     batch_size,
                     epochs,
                     model_dir)

        assert image_size[0]==image_size[1], "."

        size = 28 if self.input_width % 28 == 0 else 32 if self.input_width % 32 == 0 else False
        assert size, " [@] "
        self.K = int(self.input_width/size)*4
        self.D = 4+int(self.input_width/size)*6

    def build_model(self):

        self.X = tf.placeholder(tf.float32, [None]+self.image_size, name="input")
        self.Y = tf.placeholder(tf.float32, [None]+[self.classes], name='label')

        self.y_logit, self.y_pred = self.classifier()

        self.loss = cross_entropy(self.Y, self.y_logit)
        self.acc = get_accuracy(self.Y, self.y_logit)
        self.prediction = tf.argmax(self.y_pred, axis=1)
        self.build_optimzier(self.loss)

        self.saver = tf.train.Saver()

    def classifier(self, reuse=False):
        assert (self.D-4) % 6 == 0, " [@] d has to be 6n+4"
        n = int((self.D-4)/6)
        k = self.K
        conv2d = conv2d_3k
        width = [16, 32, 64]
        batch_norm4 = batch_norm(self.is_train, name='bn4')
        self.set_shape(image, True)
        with tf.variable_scope("classifier") as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = conv2d(image, 16, st=1, name='conv1')
            self.set_shape(conv1)
            conv2 = block(conv1, n, width[0], k, self.is_train, 1, name='conv2')
            self.set_shape(conv2)
            conv3 = block(conv2, n, width[1], k, self.is_train, 2, name='conv3')
            self.set_shape(conv3)
            conv4 = block(conv3, n, width[2], k, self.is_train, 2, name='conv4')
            self.set_shape(conv4)
            bn_relu = relu(batch_norm4(conv4))
            height, width = tensor_shape(bn_relu)[1:3]
            avg_pool = tf.nn.avg_pool(bn_relu, [1, height, width, 1], strides = [1,1,1,1], padding='VALID')
            self.set_shape(avg_pool)
            out = linear(avg_pool, self.y_dim, 'lin')
            self.set_shape(out)
            return out, tf.nn.softmax(out)
