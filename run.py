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
PIT_MODEL = ('logistic','svm','randomforest')
TMS_MODEL = ('lstm','gru','ae_lstm')
MODEL = {'logistic':LOGISTIC, 'res152':RESNET152, 'vgg19':VGGNET19,
         'svm':SVM, 'randomforest':RF}
         # 'lstm':None, 'gru':None, 'ae_lstm':None}
OPTIMIZER = {'adam':keras.optimizers.Adam,
             'gradient':keras.optimizers.SGD,
             'adadelta':keras.optimizers.Adadelta,
             'adagrad':keras.optimizers.Adagrad,
             'rmsprop':keras.optimizers.RMSprop}

def get_model_list(name=None):
    model_list = list(MODEL.keys())
    if name is None:
        return model_list
    else:
        assert name in model_list, "[!] There is no such model."
        return MODEL[name]
def get_data_list(name=None):
    data_list = list(OPEN_DATA.keys())
    if name is None:
        return data_list
    else:
        assert name in data_list, '[!] There is no such dataset'
        return OPEN_DATA[name]
def get_optimizer_list(name=None):
    opt_list = list(OPTIMIZER.keys())
    if name is None:
        return opt_list
    else:
        assert name in opt_list, "[!] There is no such optimizer"
        return OPTIMIZER[name]
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
        data_provider = DATA_PROVIDER(train_x = dataset['data']['train']['x'],
                                      train_y = dataset['data']['train']['y'],
                                      test_x = dataset['data']['test']['x'],
                                      test_y = dataset['data']['test']['y'],
                                      input_size = IMAGE_SIZE,
                                      data_type = dataset['data_type'],
                                      valid_split = float(dataset['valid_rate']),
                                      num_classes = int(dataset['output_size']))
        return {'classes':int(dataset['output_size']),
                'data_type':dataset['data_type'],
                'input_shape':(IMAGE_SIZE, IMAGE_SIZE, 3)}, data_provider

class Run(object):
    def __init__(self,
                phase,
                model_name,
                model_spec,
                dataset_spec,
                gpu,
                #                learning_rate,
                #                checkpoint_name,
                #                batch_size,
                #                optimizer,
                #                interval,
                #                max_epochs,
                **kargs):
        """
            **kargs : for advanced_option (NotImplemented)
        """

        # mode, model, data, gpu, checkpoint, max_epochs, batch_size, optimizer, lr, interval, random_seed
            # ['vgg19', {'dataset_name':'cifar10'}, '0', 'init', '5', '32', 'gradient','0.0001', '1', '0']#spec[1:]

        
        train = True if phase is 'Train' else False
        if train:
            learning_rate=kargs['learning_rate']
            checkpoint_name=kargs['checkpoint_name']
            batch_size=kargs['batch_size']
            optimizer=kargs['optimizer']
            interval=kargs['interval']
            max_epochs=kargs['max_epochs']
        else:
            optimizer=model_spec['trained']



        if gpu.isdigit() :
            print('gpu [{}] is selected'.format(gpu))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        else:
            #TODO
            print('cpu is selected')


        opt = select_optimizer(optimizer, float(learning_rate))
        data, provider = data_select(dataset_spec)
        load_model(model_name=model_name,
                          dataset_name=dataset_spec['name'],
                          name=checkpoint_name,
                          data=data,
                          data_provider=provider,
                          optimizer=opt,
                          epochs = int(max_epochs),
                          batch_size = int(batch_size),
                          train=train,
                          step_interval=float(interval))

def select_optimizer(optimizer='gradient', lr=1e-3):
    print(lr)
    assert optimizer in OPTIMIZER.keys(), "[!] There is no such optimizer."
    return {'opt': OPTIMIZER[optimizer](lr=lr),
            'name' : optimizer,
            'lr': lr,
            'arg':None}

def arg_inspection(train, data_type, model_name, step_interval):
    if train:
        assert step_interval >= 0. and step_interval <= 1.0, "[!] Ratio error : [0, 1] is allowed"
    assert data_type in DATA_TYPE, "[!] Image, Point, Time-series are only allowed."
    assert model_name is not None, "[!] Must enter the model_name"
    if data_type == 'I': # image
        assert model_name in IMG_MODEL, "[!]{} is not for {}".format(model_name, data_type)
    elif data_type == 'P': # Point data
        assert model_name in PIT_MODEL, "[!]{} is not for {}".format(model_name, data_type)
    else: # Time-series data
        assert model_name in TMS_MODEL, "[!]{} is not for {}".format(model_name, data_type)

def load_model(
               model_name,          # model name
               name                 = None,  # additional_name
               dataset_name         = None,  # dataset name, open data e.g. mnist or hand-crafted dataset
               data_provider        = None, # if own dataset is given, flag it
               data                 = None,  # dictionary : {train_x:-, train_y:-, test_x:-, test_y:-(optional)}
               optimizer            = {'opt': keras.optimizers.SGD(lr=1e-3),
                                       'name' : 'gradient',
                                       'lr': 1e-3,
                                       'arg':None},
               visualize            = True,
               batch_size           = 64,
               epochs               = 20,
               checkpoint_dir       = "checkpoint",
               checkpoint_name      = None,
               train                = False,    # if True, Do train
               epoch_interval       = 5,     # save interval
               step_interval        = 0.1,     # obtaining state rate per epoch
               pre_trained          = {'init':True,'freeze':False}      # by imagenet
               ):
    # 1. validation model
    arg_inspection(train, data['data_type'], model_name, step_interval)
    # 2. call the instance of the network
    model = MODEL[model_name](dataset_name=dataset_name,
                              num_classes=data['classes'],
                              pretrained=pre_trained['init'],
                              optimizer=optimizer,
                              batch_size = batch_size,
                              checkpoint_dir=checkpoint_dir,
                              freeze_pretrained = pre_trained['freeze'],
                              input_shape=data['input_shape'],
                              name = name
                              )
    # 3. check and load the specified model
    if not (train or test): # model_meta
        print(model())
    # 6.2 train the model.
    if train:
        if data_provider is None:
            model.train(x=data['train_x'],
                        y=data['train_y'],
                        epochs=epochs,
                        period=epoch_interval,
                        step_interval=step_interval,
                        save=True)
        else:
            g, steps = data_provider('train', batch_size)
            g_, steps_ = data_provider('valid', batch_size)
            model.train_with_generator(generator = g,
                                      valid_generator = g_,
                                      steps = steps,
                                      valid_steps = steps_,
                                      epochs = epochs,
                                      period = epoch_interval,
                                      num_x = data_provider.ntrain,
                                      step_interval = step_interval,
                                      save = True)

    # 6.3 test the model.
    else:
        assert model.trained, " [@] Train model first."
        if data_provider is None:
            model.test(x = data['test_x'],
                       y = data['test_y'],
                       label_name = data['label'],
                       visualize=visualize)
        else:
            g, steps = data_provider('test', batch_size)
            model.test_with_generator(generator = g,
                                     steps = steps,
                                     label_name = data['label'],
                                     visualize  = visualize)
