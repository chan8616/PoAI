"""
    ops.py
    Operations
"""

import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.path.abspath('..'))

from utils.data import *
from utils.util import *

FILE_FORMAT = ['jpg', 'jpeg', 'png', 'bmp']

OPTIMIZER = {'adam':tf.train.AdamOptimizer,
             'gradient':tf.train.GradientDescentOptimizer}


def get_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred, 1)), tf.float32))

def data_input(data, istrain, label=None):
    """
        Args
        - data : data(numpy array) or directory with data(str).
        - label : (optional)
        Return
        - [data] [file_path] [label(option)] [image_size] # we assume that label is not one-hot code configuration.
    """

    if type(data) == str: # directory
        print("dir2label")
        file_path, label = dir2label(data, istrain)
        print('filepath:', file_path)
        print('cwd:', os.getcwd())
        data = None
        image_size = list(image_load(file_path[0]).shape)
    elif type(data) == np.ndarray:
        data = data if len(data.shape) == 4 else data[:,:,:,np.newaxis]
        if type(label) == str:
            file_path, label = txt2label(label, istrain)
        else:
            file_path = None
        image_size = data.shape
    else:
        aassert (False, "The data has to be data itself(numpy array) or directory(string) containing data. But {}".format(type(data)))
    # pprint(label)
    if label is not None and np.max(label) > 0:
        label = one_hot_coding(label)
    return data, file_path, label, image_size

def txt2label(file_path, istrain):
    """
        Return
        - [file_name], [:[class]] : if data is located in own class directory.
        - None, [:[class]] : if all data is aggregated in a directory.
    """
    label, line = [], 0
    could = os.path.exists(file_path)
    if not could and not istrain:
        return None, None
    aassert(could or not istrain)

    with open(file_path, 'r') as f:
        while True:
            instance = f.readline().split('\n')[0].replace('\t', ' ').split(' ')
            instance = [int(i) if i.isdigit() else i for i in instance if i]
            if not instance: break
            label.append(instance)
            aassert (len(label[line]) == len(label[line-1]), "inconsistency detected on {}th line.".format(line))
            line += 1
    # line = len(label[0])
    print("label", label)
    print(file_path)
    classes_path = os.path.join('/', *file_path.split('/')[:-1], 'classes')
    if istrain:
        # classes = sorted(list(set([lb[i] for lb in label ]) for i in range(line)))
        classes = sorted(list(set([lb[1] for lb in label])))
        pickle_save(classes, classes_path)
    else:
        classes = pickle_load(classes_path)
    # str2cls = [{v:i for i, v in enumerate(cls)} for cls in classes]
    # cls2str = [{i:v for i, v in enumerate(cls)} for cls in classes]
    str2cls = {v:i for i, v in enumerate(classes)}
    cls2str = {i:v for i, v in enumerate(classes)}

    if line == 1:
        return None, np.array([str2cls[instance] for instance in label]), cls2str
    elif line >= 2:
        return [instance[0] for instance in label], np.array([str2cls[instance[1]] for instance in label])


def dir2label(dataset, istrain, file_format=FILE_FORMAT):
    """
        Return
        - [file_name], [class]
    """
    mode = 'train' if istrain else 'test'
    directory = os.path.join(os.getcwd(),'dataset', dataset)#, 'Data')
    classes = sorted([dir for dir in os.listdir(directory) if len(dir.split('.'))==1])

    if len(classes) == 1:#0: # data is aggregated.
        txt_path = os.path.join(directory, '{}.txt'.format(mode))
        aassert (os.path.exists(txt_path), " [@] %s.txt doesn't exist in %s"%(mode, txt_path))
        print("txt2label")
        return txt2label(txt_path, istrain)

    else:
        print("class", classes)
        classes_path = os.path.join(directory, 'classes')
        if istrain:
            classes = sorted([dir for dir in os.listdir(directory) if len(dir.split('.'))==1])
            pickle_save(classes, classes_path)
        else:
            classes = pickle_load(classes_path)
        file_path = [[os.path.join(directory, cls, file_name) for file_name in os.listdir(os.path.join(directory,cls)) if file_name.split('.')[-1].lower() in file_format] for cls in classes]
        label = np.concatenate([np.array([integer] * len(file_path[integer])) for integer, cls_name in enumerate(classes)])

        return sorted(list(set().union(*file_path))), label

def label2txt(label, txt_name, dataset, file_name=None):
    """
        Return :
        - [file_name] [class]\n # for each instance if file_name exists
        - [class]\n # for each instance if file_name does not exists (It implies that data has been ordered by names. )
    """
    directory = os.path.join(os.getcwd(),'dataset', dataset, 'Data')
    txt_path = os.path.join(directory, '{}.txt'.format(txt_name))
    with open(txt_path, 'w') as f:
        if file_name:
            aassert (len(label) == len(file_name))
            for i in range(len(label)):
                f.write('{} {}\n'.format(file_name[i], label[i])) # TODO : multi-label
        else:
            for i in range(len(label)):
                f.write('{}\n'.format(label[i]))
    return txt_name

def one_hot_coding(y):
    y= np.array(y)
    N = y.shape[0]
    K = np.max(y, axis=0).astype(np.int32)+1 # multiple labels can be accepted.
    one_hot = np.zeros([N, K])
    one_hot[np.arange(N), y] = 1
    return one_hot

def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def linear(x, out_dim, name, stddev=0.02):
    in_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [in_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.zeros_initializer)
    return tf.nn.bias_add(tf.matmul(x, w), b)

class batch_norm(object):
  def __init__(self, train, epsilon=1e-5, momentum = 0.9, name="batch_norm", use=True):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name
      self.train = train
      self.use = use
  def __call__(self, x):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=self.train,
                      scope=self.name) if self.use else tf.identity(x, name="no_batch")


def conv2d_3k(input_, output_dim, st=1, name='conv2d_3k'):
    return conv2d(input_, output_dim, ks=3, st=st, name=name)

def conv2d(input_, output_dim,
       ks=3, st=2, stddev=0.02, padding='SAME',name="conv2d", kw=None):
  if kw is None:
      kw, kh = ks, ks
  else:
      kw, kh = kw, ks
  with tf.variable_scope(name):
    w = tf.get_variable('w', [kw,kh, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, st, st, 1], padding=padding)

    biases = tf.get_variable('biases', [1,1,1,output_dim], initializer=tf.constant_initializer(0.0))
    return conv+biases

def block(x, n, channel, k, train, stride, name, use_batch=True):

    conv2d = conv2d_3k # alias
    out_dim = k*channel
    for i in range(n):
        x_channel = tensor_shape(x)[-1]
        with tf.variable_scope("{}_{}".format(name, i)):
            bn1 = batch_norm(train, name='bn1', use=use_batch)
            bn2 = batch_norm(train, name='bn2', use=use_batch)
            relu_bn1 = relu(bn1(x))
            conv1 = conv2d(relu_bn1, out_dim, st = stride if i==0 else 1, name='conv1')
            relu_bn2 = relu(bn2(conv1))
            conv2 = conv2d(relu_bn2, out_dim, st = 1, name='conv2')
            shortcut = x if x_channel == out_dim else conv2d(relu_bn1, out_dim, stride, name='shortcut')
            x = conv2 + shortcut
    return x
