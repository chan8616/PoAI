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
OPTIMIZER = {'adam':keras.optimizers.Adam,
             'gradient':keras.optimizers.SGD,
             'adadelta':keras.optimizers.Adadelta,
             'adagrad':keras.optimizers.Adagrad,
             'rmsprop':keras.optimizers.RMSprop}

def get_model_list():
    return MODEL

def get_data_list():
    return OPEN_DATA
def get_optimizer_lisT():
    return OPTIMIZER
# TODO:

"""
    MACRO
"""
DATA_TYPE = ('I', 'P', "T") # Image, Point, Time-series
IMAGE_SIZE = 224      # Fixed
def data_select(dataset):
    if dataset['name'] in OPEN_DATA.keys(): # data is provided
        return OPEN_DATA[dataset['name']](), None
    else:    # own dataset
        print(dataset['data']['train'][:,1])
        print(dataset['data']['test'][:,1])
        data_provider = DATA_PROVIDER(train_x = dataset['data']['train'][:,0],
                                      train_y = one_hot(dataset['data']['train'][:,1]),
                                      test_x = dataset['data']['test'][:,0],
                                      test_y = dataset['data']['test'][:,1],
                                      intput_size = IMAGE_SIZE,
                                      data_type = dataset['data_type'],
                                      valid_split = dataset['valid_rate'])
        return {'classes':dataset['output_size']}, data_provider

class Run(object):
    def __init__(self,
                phase,
                model_name,
                gpu,
                learning_rate,
                checkpoint_name,
                batch_size,
                optimizer,
                interval,
                max_epochs,
                dataset_spec,
                **kargs):
        """
            **kargs : for advanced_option (NotImplemented)
        """
        # mode, model, data, gpu, checkpoint, max_epochs, batch_size, optimizer, lr, interval, random_seed
        self.model_name = model_name
            # ['vgg19', {'dataset_name':'cifar10'}, '0', 'init', '5', '32', 'gradient','0.0001', '1', '0']#spec[1:]

        self.data_type = 'I' #TODO
        self.classes = 10  #TODO

        if gpu.isdigit() :
            print('gpu [{}] is selected'.format(gpu))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        else:
            #TODO
            print('cpu is selected')

        train = True if phase else False

        opt = self.select_optimizer(optimizer, float(learning_rate))
        data, provider = data_select(dataset_spec)
        self.load_model(model_name=self.model_name,
                          dataset_name=dataset_spec['name'],
                          name=checkpoint_name,
                          classes=self.classes,
                          data=data,
                          data_provider=provider,
                          optimizer=opt,
                          epochs = int(max_epochs),
                          batch_size = int(batch_size),
                          train=train,
                          epoch_interval=int(interval))
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
                   data_provider        = False, # if own dataset is given, flag it
                   data                 = None,  # dictionary : {train_x:-, train_y:-, test_x:-, test_y:-(optional)}
                   optimizer            = {'opt': keras.optimizers.SGD(lr=1e-3),
                                           'name' : 'gradient',
                                           'learning_rate': 1e-3,
                                           'arg':None},
                   visualize            = True,
                   batch_size           = 64,
                   epochs               = 20,
                   checkpoint_dir       = "checkpoint",
                   checkpoint_name      = None,
                   train                = False,    # if True, Do train
                   epoch_interval       = None,     # save interval
                   step_interval        = 0.1,     # obtaining state rate per epoch
                   pre_trained          = {'init':True,'freeze':False}      # by imagenet
                   ):
        # 1. validation model
        self.model_validation()
        # 2. call the instance of the network
        model = MODEL[model_name](dataset_name=dataset_name,
                                  num_classes=data['classes'],
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
            if data is not None:
                model.train(x=data['train_x'],
                            y=data['train_y'],
                            epochs=epochs,
                            period=epoch_interval,
                            step_interval=step_interval,
                            save=True)
            elif provider is not None:
                g, steps = provider('train', batch_size)
                g_, steps_ = provider('valid', batch_size)
                model.train_with_provider(generator = g,
                                          valid_generator = g_,
                                          steps = steps,
                                          valid_steps = steps_,
                                          epochs = epochs,
                                          period = epochs_interval,
                                          num_x = provider.ntrain,
                                          step_interval = step_interval,
                                          save = True)

        # 6.3 test the model.
        else:
            assert model.trained, " [@] Train model first."
            if data is not None:
                model.test(x = data['test_x'],
                           y = data['test_y'],
                           label_name = data['label'],
                           visualize=visualize)
            elif provider is not None:
                g, steps = provider('test', batch_size)
                model.test_with_provider(generator = g,
                                         steps = steps,
                                         label_name = data['label'],
                                         visualize  = visualize)
