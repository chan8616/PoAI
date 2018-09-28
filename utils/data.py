"""
    data.py
    Open data
"""


from .util import shuffle, image_load
import numpy as np
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

def call_mnist(one_hot_coding=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/mnist", one_hot=False)
    train_data, train_label = shuffle(np.concatenate((mnist.train.images, mnist.validation.images),axis=0),
                                     np.concatenate((mnist.train.labels, mnist.validation.labels),axis=0))
    test_data, test_label = shuffle(mnist.test.images, mnist.test.labels)
    if one_hot_coding:
        train_label = one_hot(train_label)
    return {'train_x':train_data.reshape(-1, 28,28,1),
            'train_y':train_label,
            'test_x': test_data.reshape(-1, 28,28,1),
            'test_y': test_label,
            'label': list(np.arange(10))}

def call_cifar10(one_hot_coding=True):
    from tensorflow.python.keras.datasets.cifar10 import load_data
    (train_data, train_label), (test_data, test_label) = load_data()
    if one_hot_coding:
        train_label = one_hot(train_label)
    train_data, train_label = shuffle(train_data, train_label)
    return {'train_x' : train_data,
            'train_y' : train_label,
            'test_x' : test_data,
            'test_y' : test_label,
            'label' : ['airplane',
                       'automobile',
                       'bird',
                       'cat',
                       'deer',
                       'dog',
                       'frog',
                       'horse',
                       'ship',
                       'truck']}

def call_wine(one_hot_coding=True, train_ratio=0.7):
    from sklearn.datasets import load_wine
    dataset = load_wine()
    data = dataset['data']
    label = dataset['target']
    data, label = shuffle(data, label)
    train = int(len(data)*train_ratio)
    train_data = data[:train]
    train_label = one_hot(label[:train]) if one_hot_coding else label[:train]
    test_data = data[train:]
    test_label = label[train:]

    return {'train_x':train_data,
            'train_y':train_label,
            'test_x':test_data,
            'test_y':test_label,
            'label':[list(np.arange(3))]}

def call_iris(one_hot_coding=True, train_ratio=0.7):
    from sklearn.datasets import load_iris
    dataset = load_iris()
    data = dataset['data']
    label = dataset['target']
    data, label = shuffle(data, label)
    train = int(len(data)*train_ratio)
    train_data = data[:train]
    train_label = one_hot(label[:train]) if one_hot_coding else label[:train]
    test_data = data[train:]
    test_label = label[train:]

    return {'train_x':train_data,
            'train_y':train_label,
            'test_x':test_data,
            'test_y':test_label,
            'label':['Setosa', 'Versicolour', 'Virginica']}


class DATA_PROVIDER(object):
    def __init__(self,
                 train_x,
                 train_y,
                 test_x,
                 test_y = None,
                 valid_x = None,
                 valid_y = None,
                 label_info = None,
                 input_size = 224, # if zero, do not resize
                 shuffle=True,
                 data_type='I',
                 valid_split = 0.0):
        self.x, self.y, self.is_file, self.data_type, self.input_size = {}, {}, {}, data_type, input_size
        self.label_info = label_info
        if shuffle:
            train_x, train_y = shuffle(train_x, train_y)

        if valid_x is not None and valid_y is not None:
            self.x['valid'] = valid_x
            self.y['valid'] = valid_y
            self.is_file['valid'] = True if type(valid_x[0]) == str else False

            valid = 0
        elif valid_split > 0.:
            assert valid_split < 1.0
            valid = int(len(train_x)*valid_split)
            self.x['valid'] = train_x[:valid]
            self.y['valid'] = train_x[:valid]
            self.is_file['valid'] = True if type(train_x[0]) == str else False

        self.x['train'] = train_x[valid:]
        self.y['train'] = train_y[valid:]
        self.x['test'] = test_x
        self.y['test'] = test_y

        # inspecting data
        # data is given as file path #TODO : what if the data is some string?
        self.is_file['train'] = True if type(self.train_x[0]) == str else False
        self.is_file['test'] = True if type(self.test_x[0]) == str else False
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
        assert mode in ['train', 'test'], "[!] Only 'train' or 'test' is allowed."
        x = self.x[mode]
        batch_size = batch_size if batch_size else len(x)
        steps = np.ceil(len(x)/batch_size)
        if self.is_file[mode]:
            return lambda :self.generate_arrays_from_file(mode, batch_size, steps), steps
        else:
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
        length = len(data_list)
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
