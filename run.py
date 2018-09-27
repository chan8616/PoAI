import tensorflow as tf
import numpy as np
import os

from utils.data import *

from tensorflow import keras
"""
    model list
"""

from model.simple import LOGISTIC # simple classifier
from model.vgg19 import VGGNET19
from model.resnet import RESNET152
from model.svm import SVM
from model.rf import RF
#TODO:
# from model.resnet import RES #
# from model.lstm import LSTM
# from model.gru import GRU
# from model.lstm import AELSTM

"""
    dataset list
"""
OPEN_IMG = ('mnist', 'cifar10')
OPEN_PIT = ('wine', 'iris')
OPEN_TMS = ('bearing')
OPEN_DATA = {'mnist':call_mnist,
             'cifar10':call_cifar10,
             'wine':call_wine,
             'iris':call_iris}
# TODO: one time-series data
"""
    model list
"""
IMG_MODEL = ('logistic', 'res152', 'vgg19')
PIT_MODEL = ('svm','randomforest')
TMS_MODEL = ('lstm','gru','ae_lstm')
MODEL = {'logistic':LOGISTIC, 'res152':RESNET152, 'vgg19':VGGNET19,
         'svm':SVM, 'randomforest':RF,
         'lstm':None, 'gru':None, 'ae_lstm':None}

def get_model_list():
    return MODEL

def get_data_list():
    return OPEN_DATA

# TODO:

"""
    MACRO
"""
OPTIMIZER = {'adam':keras.optimizers.Adam,
             'gradient':keras.optimizers.SGD,
             'adadelta':keras.optimizers.Adadelta,
             'adagrad':keras.optimizers.Adagrad,
             'rmsprop':keras.optimizers.RMSprop}
DATA_TYPE = ('I', 'P', "T") # Image, Point, Time-series
IMAGE_SIZE = (224,224)      # Fixed
def data_select(dataset_name):
    if dataset_name in OPEN_DATA.keys(): # data is provided
        return OPEN_DATA[dataset_name](), None
    else:    # own dataset
        return None, None

class Run(object):
    def __init__(self, **kargs):
        """
            **kargs : for advanced_option (NotImplemented)
        """
        # mode, model, data, gpu, checkpoint, max_epochs, batch_size, optimizer, lr, interval, random_seed
        print(kargs)
        print(get_data_list())
        print(get_model_list())
        self.model_name, dataset_name, gpu_selected, \
        name, epochs, batch_size, optimizer, \
        learning_rate, interval, random_seed =\
            ['vgg19', 'cifar10', '0', 'init', '5', '32', 'gradient','0.0001', '1', '0']#spec[1:]

        gpu_selected = int(gpu_selected)
        self.epochs = int(epochs)
        batch_size = int(batch_size)
        interval = int(interval)

        self.data_type = 'I' #TODO
        self.classes = 10  #TODO

        if gpu_selected.isdigit() :
            print('gpu [{}] is selected'.format(gpu_selected))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_selected)
        else:
            print('cpu is selected')

        self.train = True if 'train' in spec[0] else False
        self.test = True if 'test' in spec[0] else False

        optimizer = self.select_optimizer(optimizer, float(learning_rate))
        data, provider = data_select(dataset_name)
        self.load_model(model_name=self.model_name,
                          dataset_name=dataset_name,
                          name=name,
                          classes=self.classes,
                          data=data,
                          data_provider=provider,
                          optimizer=optimizer,
                          epochs = self.epochs,
                          batch_size = batch_size,
                          train=self.train,
                          test=self.test,
                          epoch_interval=interval,
                          random_seed=random_seed)
    def select_optimizer(self, optimizer='gradient', lr=1e-3):
        assert optimizer in OPTIMIZER.keys(), "[!] There is no such optimizer."
        return {'opt': OPTIMIZER[optimizer](lr=lr),
                'name' : optimizer,
                'lr': lr,
                'arg':None}

    def model_validation(self):
        assert self.data_type in DATA_TYPE, "Image, Point, Time-series are only allowed."
        assert self.model_name is not None, "Must enter the model_name"
        if self.data_type == 'I': # image
            assert self.model_name in IMG_MODEL, "{} is not for {}".format(self.model_name, self.data_type)
        elif self.data_type == 'P': # Point data
            assert self.model_name in PIT_MODEL, "{} is not for {}".format(self.model_name, self.data_type)
        else: # Time-series data
            assert self.model_name in TMS_MODEL, "{} is not for {}".format(self.model_name, self.data_type)

    def load_model(self,
                   model_name,          # model name
                   name                 = None,  # additional_name
                   dataset_name         = None,  # dataset name, open data e.g. mnist or hand-crafted dataset
                   classes              = None,  # We treat this model as classifier if it has values
                   data_provider        = False, # if own dataset is given, flag it
                   data                 = None,  # dictionary : {train_x:-, train_y:-, test_x:-, test_y:-(optional)}
                   optimizer            = {'opt': keras.optimizers.SGD(lr=1e-3),
                                           'name' : 'gradient',
                                           'learning_rate': 1e-3,
                                           'arg':None},
                   batch_size           = 64,
                   epochs               = 20,
                   checkpoint_dir       = "checkpoint",
                   checkpoint_name      = None,
                   train                = False,    # if True, Do train
                   test                 = False,    # if True, Do test
                   epoch_interval       = None,     # save interval
                   step_interval        = 0.1,     # obtaining state rate per epoch
                   random_seed          = 0,
                   pre_trained          = {'init':True,'freeze':False}      # by imagenet
                   ):
        # 1. validation model
        self.model_validation()
        # 2. call the instance of the network
        model = MODEL[model_name](dataset_name=dataset_name,
                                  num_classes=classes,
                                  pretrained=pre_trained['init'],
                                  optimizer=optimizer,
                                  batch_size = batch_size,
                                  checkpoint_dir=checkpoint_dir,
                                  freeze_pretrained = pre_trained['freeze'],
                                  name = name
                                  )
        # 3. check and load the specified model
        if not (train or test): # model_meta
            print(model())
        # 6.2 train the model.
        if train:
            model.train(x=data['train_x'],
                        y=data['train_y'],
                        epochs=epochs,
                        period=epoch_interval,
                        step_interval=step_interval,
                        save=True)
        # 6.3 test the model.
        if test:
            assert model.trained, " [@] Train model first."
            model.test(data['test_x'], data['test_y'])
