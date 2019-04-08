import tensorflow as tf
import numpy as np
import os

from utils.data import *
from utils.util import image_load

from tensorflow import keras
"""
    model list
"""

from model.simple import LOGISTIC # simple classifier
from model.vgg import VGGNET19, VGGNET16
from model.resnet import RESNET50, RESNET101, RESNET152
from model.densenet import DENSENET121, DENSENET169, DENSENET201
from model.inception import INCEPTIONV3
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
IMG_MODEL = ('logistic',
             'resnet50',
             'resnet101',
             'resnet152',
             'densenet121',
             'densenet169',
             'densenet201',
             'inception_v3',
             'vgg16',
             'vgg19')
PIT_MODEL = ('logistic','svm','randomforest')
TMS_MODEL = ('lstm','gru','ae_lstm')
_CLASSIFICATION_ = ('logistic',
             'resnet50',
             'resnet101',
             'resnet152',
             'densenet121',
             'densenet169',
             'densenet201',
             'inception_v3',
             'vgg16',
             'vgg19',
             'logistic',
             'svm',
             'randomforest')
MODEL = {'logistic':LOGISTIC,
         'vgg16':VGGNET16,
         'vgg19':VGGNET19,
         'resnet50':RESNET50,
         'resnet101':RESNET101,
         'resnet152':RESNET152,
         'densenet121':DENSENET121,
         'densenet169':DENSENET169,
         'densenet201':DENSENET201,
         'inception_v3':INCEPTIONV3,
         'svm':SVM,
         'randomforest':RF}
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
        if name in IMG_MODEL:
            Type = 'Image'
        elif name in PIT_MODEL:
            Type = 'Point'
        elif name in TMS_MODEL:
            Type = 'Time-series'
        else:
            assert False, "[!] Not Supported"
        if name in _CLASSIFICATION_:
            Task = 'Classification'
        else:
            assert False, "[!] Not Supported"
        return {'input_type':Type, 'network_type':Task}
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
def data_select(dataset, input_size, train=True):
    try:
        dataset['name']
    except:
        test_images = dataset['data']['test']['x']
        test_x = np.array([image_load(img, input_size[0]) for img in test_images]).astype(np.float32)
        #        if input_size != test_images.shape:
        #            if (input_size[-1] == 3 and test_images.shape[-1] == 1):
        #
        #                data = {'test_x':test_x

        data = {'test_x': test_x,
                'test_y': None}
        return data, None

    if dataset['name'] in OPEN_DATA.keys(): # data is provided
        if dataset['input_types']=='Image':
            return OPEN_DATA[dataset['name']](input_shape=input_size), None
        else:
            return OPEN_DATA[dataset['name']](), None
    else:    # own dataset
        if train:
            DP = lambda : DATA_PROVIDER(train_x=dataset['data']['train']['x'],
                                        train_y=dataset['data']['train']['y'],
                                        input_size = input_size[0],
                                        data_type = dataset['data_type'],
                                        valid_split = float(dataset['valid_rate']),
                                        num_classes=int(dataset['output_size']),
                                        train=train)
            spec['classes'] = int(dataset['output_size'])
        else:
            try:
                dataset['data']['test']['y']
            except NameError:
                dataset['data']['test']['y'] = None
            DP = lambda x, y: DATA_PROVIDER(test_x=dataset['data']['test']['x'],
                                            test_y=dataset['data']['test']['y'],
                                            input_size = input_size[0],
                                            data_type=dataset['data_type'],
                                            train=train)
        data_provider = DP()
        return None, data_provider

class Run(object):
    def __init__(self,
                phase,
                model_name,
                model_spec,
                dataset_spec,
                gpu,
                learning_rate = 1e-3,
                checkpoint_name = '',
                #                batch_size,
                #                optimizer,
                #                interval,
                #                max_epochs,
                **kargs):
        """
            **kargs : for advanced_option (NotImplemented)
        """

        is_train = True if phase is 'Train' else False
        is_trained = True if 'trained' in model_spec.keys() else False
        # print(kargs)
        # print(dataset_spec)
        # assert False
        # data, provider = data_select(dataset_spec, train)
        if gpu.isdigit() :
            print('gpu [{}] is selected'.format(gpu))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        else:
            print('cpu is selected')
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        if is_train:
            interval=float(kargs['interval'])
            max_epochs=int(kargs['max_epochs'])
            if is_trained:
                model_ckpt = model_spec['trained']['ckpt_path']
                load_model(model_ckpt = model_ckpt,
                           model_name = model_name,
                           curr_data = dataset_spec,
                           epochs = max_epochs,
                           step_interval = interval,
                           is_train = is_train)
            else:
                batch_size=int(kargs['batch_size'])
                optimizer=kargs['optimizer']
##*
                print(model_name, dataset_spec, checkpoint_name, optimizer, learning_rate, max_epochs, batch_size, is_train, interval)
                load_model(model_name = model_name,
                           base_data_spec = dataset_spec,
                           curr_data = dataset_spec,
                           name = checkpoint_name,
                           opt_arg = [optimizer, float(learning_rate)],
                           epochs = max_epochs,
                           batch_size = batch_size,
                           is_train = is_train,
                           step_interval = interval)
        else:
            assert is_trained, "[!] train first"
            model_ckpt = model_spec['trained']['ckpt_path']
            load_model(model_ckpt = model_ckpt,
                       model_name = model_name,
                       curr_data = dataset_spec,
                       is_train = is_train)


def select_optimizer(optimizer='gradient', lr=1e-3):
    if optimizer in OPTIMIZER.keys():
        return {'opt': OPTIMIZER[optimizer](lr=lr),
                'name' : optimizer,
                'lr': lr,
                'arg':None}
    else:
        return {'opt': keras.optimizers.SGD(lr=1e-3),
                'name' : 'gradient',
                'lr': lr,
                'arg':None}

def arg_inspection(train, data_type, model_name, step_interval):
    if train:
        assert step_interval >= 0. and step_interval <= 1.0, "[!] Ratio error : [0, 1] is allowed"
    assert data_type in DATA_TYPE, "[!] Image, Point, Time-series are only allowed."
    assert model_name is not None, "[!] Must enter the model_name"
    if data_type == 'I': # image
        assert model_name in IMG_MODEL, "[!] {} is not for {}".format(model_name, data_type)
    elif data_type == 'P': # Point data
        assert model_name in PIT_MODEL, "[!] {} is not for {}".format(model_name, data_type)
    else: # Time-series data
        assert model_name in TMS_MODEL, "[!] {} is not for {}".format(model_name, data_type)

def load_model(
               curr_data,           # input data
               model_name           = None, # model name
               base_data_spec       = None, # base dataset spec
               name                 = None,  # additional_name
               # dataset_name         = None,  # dataset name, open data e.g. mnist or hand-crafted dataset
               # data_provider        = None, # if own dataset is given, flag it
               # data                 = None,  # dictionary : {train_x:-, train_y:-, test_x:-, test_y:-(optional)}
               opt_arg              = None,  # optimizer arguments [optimizer name, learning_rate]
               visualize            = False,
               batch_size           = 64,
               epochs               = 20,
               checkpoint_dir       = "checkpoint",
               checkpoint_name      = None,
               is_train             = False,    # if True, Do train
               epoch_interval       = 5,     # save interval
               step_interval        = 1.0,     # obtaining state rate per epoch
               pre_trained          = {'init':True,'freeze':False},      # by imagenet
               model_ckpt           = None # if trained.
               ):


    # 0. set an optimizer
    try:
        optimizer = select_optimizer(opt_arg[0], float(opt_arg[1]))
    except:
        optimizer = select_optimizer()

    # 1. validation model
    arg_inspection(is_train, curr_data['data_type'], model_name, step_interval)

    # 2. call the instance of the network
    if model_ckpt:
        model = MODEL[model_name](model_ckpt=model_ckpt)
    else:
        model = MODEL[model_name](dataset_name=base_data_spec['name'], #
                                  num_classes=int(base_data_spec['output_size']),
                                  pretrained=pre_trained['init'],
                                  optimizer=optimizer,
                                  batch_size = batch_size,
                                  checkpoint_dir=checkpoint_dir, #
                                  freeze_pretrained=pre_trained['freeze'],
                                  input_shape=base_data_spec['input_shape'],
                                  name=name,
                                  label_names=curr_data['label_names'])

    # 3. arrange dataset
    try:
        data, data_provider = data_select(curr_data, model.image_shape, is_train)
    except:
        if curr_data['name'] in OPEN_DATA.keys(): # data is provided
            data, data_provider = OPEN_DATA[curr_data['name']](), None
        else:
            raise "Not Yet Implemented"

    # 6.2 train the model.
    if is_train:
        if data_provider is None:
            model.train(x=data['train_x'],
                        y=data['train_y'],
                        batch_size=batch_size,
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
        assert model.trained, " [@] Train the model first (Recommend removing the model checkpoint dir)."
        if data_provider is None:
            model.test(x = data['test_x'],
                       y = data['test_y'],
                       label_name = model.model_conf['label_names'],
                       visualize=visualize)
        else:
            g, steps = data_provider('test', batch_size)
            model.test_with_generator(generator = g,
                                     steps = steps,
                                     # label_name = data['label_names'],
                                     visualize  = visualize)
