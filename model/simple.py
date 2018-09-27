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

class LOGISTIC(NET):
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

        super(LOGISTIC, self).__init__(sess,
                     model_name,
                     dataset,
                     learning_rate,
                     optimizer,
                     beta1,
                     beta2,
                     batch_size,
                     epochs,
                     model_dir)

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
        im_dim = np.product(np.array(self.image_size))
        with tf.variable_scope('classifier') as scope:
            if reuse:
                scope.reuse_variables()
            X = self.X if len(self.X.get_shape().as_list())==2 else tf.reshape(self.X, [-1, im_dim])
            h = linear(X, self.classes, 'linear') # TODO : multi-classes
        return h, tf.nn.softmax(h)
