"""
    data.py
    Open data
"""

from .util import shuffle as _shuffle
from .util import image_load
from .util import resize_data
from tensorflow import keras
import numpy as np
import cv2

import sklearn

def one_hot(y, classes=None):

    if len(y.shape) > 2:
        return y

    if classes is None:
        classes = np.max(y)+1
    y = y.reshape(-1)
    N = y.shape[0]
    y_one_hot = np.zeros((N, classes))
    y_one_hot[np.arange(N),y]=1
    return y_one_hot

def call_mnist(meta=False):
    from tensorflow.python.keras.datasets.mnist import load_data
    (train_data, train_label), (test_data, test_label) = load_data()#path='/tmp/mnist.npz') from window, cause error.
    train_data, train_label = _shuffle(train_data, train_label)
    if meta:
        return {'ntrain':len(train_data),
                'ntest':len(test_data),
                'classes':10,
                'label_names': list(np.arange(10)),
                'input_shape': (28, 28, 1),
                'data_type':'I'}
    else:
        return {'train_x':train_data.reshape(-1, 28,28,1),
                'train_y':train_label,
                'test_x': test_data.reshape(-1, 28,28,1),
                'test_y': test_label,
                'classes':10,
                'label_names': list(np.arange(10)),
                'input_shape': (28, 28, 1),
                'data_type':'I'}

def call_cifar10(meta=False, input_shape=None):
    from tensorflow.python.keras.datasets.cifar10 import load_data
    from tensorflow.python.keras.utils import to_categorical
    (train_data, train_label), (test_data, test_label) = load_data()
    if meta:
        print("meta is true")
        print(meta)
        return {'ntrain':len(train_data),
                'ntest':len(test_data),
                'classes':10,
                'label_names' : ['airplane',
                               'automobile',
                               'bird',
                               'cat',
                               'deer',
                               'dog',
                               'frog',
                               'horse',
                               'ship',
                               'truck'],
                'input_shape': (32, 32, 3),
                'data_type':'I'}
    else:
        if input_shape:
            train_data = np.array(train_data).astype(float)
            test_data = np.array(test_data).astype(float)
            train_data = resize_data(train_data, input_shape)
            test_data = resize_data(test_data, input_shape)
            train_label = to_categorical(train_label, 10)
            test_label = to_categorical(test_label, 10)
        return {'train_x' : train_data,
                'train_y' : train_label,
                'test_x' : test_data,
                'test_y' : test_label,
                'classes':10,
                'label_names' : ['airplane',
                               'automobile',
                               'bird',
                               'cat',
                               'deer',
                               'dog',
                               'frog',
                               'horse',
                               'ship',
                               'truck'],
                'input_shape': tuple(input_shape),
                'data_type':'I'}


def call_wine(train_ratio=0.7, meta=False):
    from sklearn.datasets import load_wine
    dataset = load_wine()
    data = dataset['data']
    label = dataset['target']
    data, label = _shuffle(data, label)
    train = int(len(data)*train_ratio)
    train_data = data[:train]
    train_label = label[:train]
    test_data = data[train:]
    test_label = label[train:]
    if meta:
        return {'ntrain':len(train_data),
                'ntest':len(test_data),
                'classes':3,
                'label_names':[list(np.arange(3))],
                'input_shape':(13,),
                'data_type':'P'
                }
    else:
        return {'train_x':train_data,
                'train_y':train_label,
                'test_x':test_data,
                'test_y':test_label,
                'classes':3,
                'label_names':[list(np.arange(3))],
                'input_shape':(13,),
                'data_type':'P'
                }

def call_iris(train_ratio=0.7, meta=False):
    from sklearn.datasets import load_iris
    dataset = load_iris()
    data = dataset['data']
    label = dataset['target']
    data, label = _shuffle(data, label)
    train = int(len(data)*train_ratio)
    train_data = data[:train]
    print(train_data.shape)
    train_label = label[:train]
    test_data = data[train:]
    test_label = label[train:]
    if meta:
        return {'ntrain':len(train_data),
                'ntest':len(test_data),
                'classes':3,
                'label_names':['Setosa', 'Versicolour', 'Virginica'],
                'input_shape':(4,),
                'data_type':'P'}
    else:
        return {'train_x':train_data,
                'train_y':train_label,
                'test_x':test_data,
                'test_y':test_label,
                'classes':3,
                'label_names':['Setosa', 'Versicolour', 'Virginica'],
                'input_shape':(4,),
                'data_type':'P'}


class DATA_PROVIDER(object):
    def __init__(self,
                 train_x=None,
                 train_y=None,
                 test_x=None,
                 test_y = None,
                 train=True,
                 valid_x = None,
                 valid_y = None,
                 label_info = None,
                 input_size = 224, # if zero, do not resize
                 shuffle=True,
                 data_type='I',
                 num_classes=None,
                 valid_split = 0.0):
        self.x, self.y, self.is_file, self.data_type, self.input_size = {}, {}, {}, data_type, input_size
        self.label_info = label_info
        if train:
            if shuffle:
                train_x, train_y = _shuffle(train_x, train_y)

            if valid_x is not None and valid_y is not None:
                self.x['valid'] = valid_x
                self.y['valid'] = one_hot(valid_y, classes=num_classes)
                self.is_file['valid'] = True if type(valid_x[0]) == str else False

                valid = 0
            elif valid_split > 0.:
                assert valid_split < 1.0
                valid = int(len(train_x)*valid_split)
                self.x['valid'] = train_x[:valid]
                self.y['valid'] = one_hot(train_y[:valid], classes=num_classes)
                self.is_file['valid'] = True if type(train_x[0]) == np.str_ else False
            print(train_y)
            self.x['train'] = train_x[valid:]
            self.y['train'] = one_hot(train_y[valid:], classes=num_classes)
            self.is_file['train'] = True if type(self.x['train'][0]) == np.str_ else False
            print(self.is_file, type(self.x['train'][0]))
            self.x['test'] = test_x
            self.y['test'] = test_y
            self.is_file['test'] = True if type(self.x['test'][0]) == np.str_ else False
        else:
            assert test_x is not None, "[!] test data"
            self.x['test'] = test_x
            self.y['test'] = test_y
            self.is_file['test'] = True if type(self.x['test'][0]) == np.str_ else False

        # inspecting data
        # data is given as file path #TODO : what if the data is some string?
    @property
    def label(self):
        return self.label_info if self.label_info is not None else \
               list(np.arange(self.y['train'].shape[1]))

    @property
    def ntrain(self):
        return len(self.x['train'])

    def __call__(self,
                 mode,
                 batch_size = 100):
        assert mode in ['train', 'test', 'valid'], "[!] Only 'train' or 'test' is allowed."
        x = self.x[mode]
        batch_size = batch_size if batch_size else len(x)
        steps = int(np.ceil(len(x)/batch_size))
        if self.is_file[mode]:
            print('from file')
            return lambda :self.generate_arrays_from_file(mode, batch_size, steps), steps
        else:
            print('from mem')
            return lambda :self.generate_arrays_from_mem(mode, batch_size, steps), steps

    def generate_arrays_from_file(self, mode, batch_size, steps, with_y=True):
        data_list = self.x[mode]
        length = len(data_list)
        if with_y:
            label_list = self.y[mode]
            if self.data_type == 'I':
                while True:
                    for step in range(steps):
                        offset = step*batch_size
                        next_idx = offset+batch_size if offset+batch_size <= length else None
                        data_idx = data_list[offset:next_idx]
                        label_idx = label_list[offset:next_idx]
                        input_ = np.array([image_load(d, self.input_size) for d in data_idx]).astype(float)
                        label_ = np.array([l for l in label_idx])
                        yield (input_,label_)
            else:
                raise NotImplementedError('.')
        else:
            if self.data_type == 'I':
                while True:
                    for step in range(steps):
                        offset = step*batch_size
                        next_idx = offset+batch_size if offset+batch_size <= length else None
                        data_idx = data_list[offset:next_idx]
                        input_ = np.array([image_load(d, self.input_size) for d in data_idx]).astype(float)
                        yield (input_)
            else:
                raise NotImplementedError('.')

    def generate_arrays_from_mem(self, mode, batch_size, steps, with_y=True):
        x = self.x[mode]
        length = len(x)
        if with_y:
            y = self.y[mode]
            while True:
                for step in range(steps):
                    offset = step*batch_size
                    next_idx = offset+batch_size if offset+batch_size <= length else None
                    yield (x[offset:next_idx], y[offset:next_idx])
        else:
            while True:
                for step in range(steps):
                    offset = step*batch_size
                    next_idx = offset+batch_size if offset+batch_size <= length else None
                    yield (x[offset:next_idx])
