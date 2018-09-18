import sklearn
import tensorflow as tf
import numpy as np
import os

from utils.data import *

"""
    model list
"""

from model.simple import LOGISTIC # simple classifier
from model.wrn import WRN # wide residual net


"""
    dataset list
"""

open_data = {'mnist':call_mnist}

MODEL = {'simple':LOGISTIC, 'wrn':WRN}

def model_meta(file_name, **kargs):
    pickle_save(kargs, file_name)
    return kargs


class Run(object):
    def __init__(self, spec):
        #<<<<<<< HEAD
#        print(spec)
#        # phase, model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
#        # phase, model_spec, data_spec, train_spec
#        # model_spec: name, path, 
#        # data_spec: name, path, 
#        # train_spec: checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
#
#        # train_spec: model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
#        # test_spec: model, data, # of images, max_per_number
#        self.args = {}
#        self.args['train'] = spec[0] == 'train'
#        self.args['model'] = spec[1]
#        if self.args['train']:
#            self.args['dataset'] = spec[2]
#            self.args['checkpoint'] = spec[3]
#            self.args['max_iter'] = int(spec[4])
#            self.args['batch_size'] = int(spec[1])
#            self.args['optimizer'] = spec[1]
#            self.args['lr'] = int(spec[1])
#            self.args['interval'] = int(spec[1])
#            self.args['random_seed'] = int(spec[1])
#    
#            if self.args['model']== 'logistic':
#                self.logistic()
#        elif spec[0] == 'test':
#=======
        # mode, model, data, gpu, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
        print(spec)
        self.model_name, self.dataset_name, gpu_selected, \
        self.checkpoint, self.epochs, self.batch_size, self.optimizer, \
        self.learning_rate, self.interval, self.random_seed = spec[1:]
        gpu_selected = int(gpu_selected)


        self.data_type = 'Image' #TODO
        self.classes = 10  #TODO


        print('gpu [{}] is selected'.format(gpu_selected))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_selected)
        self.train = True if 'train' in spec[0] else False
        self.test = True if 'test' in spec[0] else False

    @property
    def meta(self):
        meta =  {'Name':self.model_name,
                 'Dataset':self.dataset_name,
                 'Data_type':self.data_type,
                 'Learning_rate':self.learning_rate,
                 'Optimizer':self.optimizer,
                 'Batch_size':self.batch_size,
                 'Random_seed':self.random_seed}
        try:
            meta['Epoch_trained'] = self.epoch_trained
        except:
            pass
        return meta

    def __call__(self, arg):
        with tf.Session() as sess:
            return load_model(sess=sess,
                              model_name=self.model_name,
                              dataset_name=self.dataset_name,
                              checkpoint_name=self.checkpoint,
                              epochs = self.epochs,
                              batch_size = self.batch_size,
                              train=self.train,
                              test=self.test,
                              step_interval=self.interval,
                              random_seed=self.random_seed)

    def load_model(self,
                   sess,               # tf.Session()
                   model_name,              # model name
                   dataset_name         = None,     # dataset name, open data e.g. mnist or hand-crafted dataset
                   classes              = None,     # We treat this model as classifier if it has values
                   image_size           = None,  # if it is given, image can be cropped or scaling
                   train_data           = None,  # folder with images
                   test_data            = None,   # folder with images
                   train_label          = None, # txt file
                   test_label           = None,  # (optional) txt file
                   learning_rate        = 1e-4,
                   optimizer            = 'gradient',
                   beta1                = 0.5,   # (optional) for some optimizer
                   beta2                = 0.99,  # (optional) for some optimizer
                   batch_size           = 64,
                   epochs               = 20,
                   checkpoint_dir       = "checkpoint",
                   checkpoint_name      = None,
                   train                = False,    # if True, Do train
                   test                 = False,    # if True, Do test
                   epoch_interval       = None,
                   step_interval        = None,
                   random_seed          = 0
                   ):

        """
            Argments
            - sess          (Essential)
            - dataset_name  (Essential)
            - model_name    (Essential)
            - optimizer     (Essential)
            - learning_rate (Essential)
            - beta1         (Essential)
            - beta2         (Essential)
            - batch_size    (Essential)
            - epoch         (Essential)
            - image_size    (Optional)
            - model_dir     (Optional)
            - classes       (Optional)
        """
        assert model_name is not None and model_name in MODEL.keys()

        arg = [sess, dataset_name, model_name, optimizer, learning_rate, beta1, beta2, batch_size, epoch, image_size, model_dir]
        arg += [classes] if classes > 0 else []
        model = MODEL[model_name](*arg)
        load, epoch_trained = net.load(checkpoint_dir, checkpoint_name)
        if not (train or test):
            if model_dir:
                pass
            return
        if train: net.train(epochs-epoch_trained, checkpoint_dir, checkpoint_name, epoch_interval, step_interval, train_data, train_label)
        if test:
            assert (train or load, " [@] Train model first.")
            net.test(test_data, test_label)
