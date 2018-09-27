import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Model

from copy import deepcopy
from os import path, makedirs

import sys
sys.path.append(path.abspath('..')) # refer to sibling folder

from .ops import *
from utils.data import *
from utils.util import *

"""
    #1 By dictionary
        Train Data path : [Project Folder]/dataset/[data name]/Data/[Class name]/[Data_name].[format]
        Test Data path : [Project Folder]/dataset/[data name]/Data/test/[Data_name].[format]
    #2 By txt
        Data path : [Project Folder]/dataset/[data name]/Data/[Data_name].[format]
    Checkpoint directory : [Project Folder]/checkpoint/[Model_name]/[Epoch or Iteration].ckpt

"""
class NET(object):
    def __init__(self,
                 dataset_name,
                 num_classes,
                 pretrained,
                 optimizer, # it contains information about optimizer
                 batch_size,
                 checkpoint_dir,
                 model,
                 freeze_pretrained,
                 init,
                 name,
                 checkpoint_name = 'model',
                 additional_layer = [1024]
                 ):

        self.add_layer= additional_layer
        # 1. Define a name and a checkpoint path of the model

        self.model_name = '{}_{}'.format(dataset_name, name) if name is not None else dataset_name
        checkpoint_dir = path.join(checkpoint_dir, model)
        if not path.exists(checkpoint_dir):
            makedirs(checkpoint_dir)
        self.model_dir = path.join(checkpoint_dir, self.model_name)
        self.model_ckpt = path.join(self.model_dir, checkpoint_name+'.h5')
        self.model_meta = path.join(self.model_dir, 'meta')
        self.num_classes = num_classes

        # 2. Check if the model is saved
        model_check = self.model_check()

        # 3. Set initialization policy
        weight_init = init[0] if pretrained and not model_check else init[1]

        # 4. Set some arguments of the models
        self.optimizer = optimizer['opt']

        # 5. Set configurations for building model and callbacks

        # 6. Build or load the model
        if model_check:
            # model_conf overwritten from the initiative.
            self.model, self.model_conf, self.epochs = self.restore()
        else:
            self.model_conf = {'name':self.model_name,
                               'model_dir':self.model_dir,
                               'ckpt_path':self.model_ckpt,
                               'meta':self.model_meta,
                               'init':weight_init,
                               'freeze':freeze_pretrained,
                               'dataset':dataset_name,
                               'batch_size':batch_size,
                               'optimizer':optimizer['name'],
                               'learning_rate':optimizer['lr'],
                               'optimizer_arg':optimizer['arg']}
            self.epochs = 0
            self.build_model(self.model_conf)
            # save model's configuration
            pickle_save(self.prog_info, self.model_meta)

    def __call__(self):
        """
            return model meta
        """
        return self.prog_info

    def model_check(self):
        """
            check whether there exists save files.
            return existence, model_checkpoint path
        """
        if not path.exists(self.model_dir):
            makedirs(self.model_dir)
            return False
        else:
            if not path.exists(self.model_ckpt):
                return False
            return True

    def build_model(self):
        raise NotImplementedError('.')

    def test(self):
        raise NotImplementedError('.')

    def merge_callbacks(self, conf):
        name, batch_size = self.model_conf['name'], self.model_conf['batch_size']
        return [ # keras.callbacks.Tensorboard(log_dir=conf['log_dir']),
            LossHistory(name, batch_size, conf['ntrain'], conf['step_interval']),
            keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=conf['patience']),
            keras.callbacks.ModelCheckpoint(filepath=conf['ckpt_path'],
                                            monitor='val_loss',
                                            save_best_only=conf['save_best_only'],
                                            period=conf['period'])]

    def restore(self):
        """
            restore model, its configuration and epochs
        """
        model = keras.models.load_model(self.model_ckpt)
        conf = pickle_load(self.model_meta)
        return model, conf, conf['epochs']

    def save(self):
        """
            save entire for loading the model and configuration for displaying
        """
        self.model.save(self.model_ckpt)
        conf = pickle_load(self.model_meta)
        conf['epochs'] = self.epochs
        pickle_save(conf, self.model_meta)

    def predict(self, x):
        return self.model.predict(x)
    def accuracy(self, x, y):
        assert y is not None
        y_pred = self.predict(x)
        return np.mean(np.equal(np.argmax(y_pred, axis=1), y).astype(np.float))
    def train_with_provider(self, generator, epochs, save=True):
        pass

    def train(self,
              x,
              y,
              epochs,
              period,
              step_interval,
              save=True):

        debug_conf = {'log_dir':path.join('./logs', self.model_name),
                      'ntrain' : len(x),
                      'save_best_only':True,
                      'ckpt_path':self.model_ckpt,
                      'step_interval':step_interval,
                      'period':period,
                      'patience':2}

        self.callbacks = self.merge_callbacks(debug_conf)
        if self.epochs >= epochs:
            print("[!] Already Done.")
            return 0
        self.model.fit(x=x,
                       y=y,
                       epochs=epochs,
                       validation_split=0.01,
                       initial_epoch=self.epochs,
                       callbacks=self.callbacks,
                       verbose=0)
        self.epochs = epochs
        if save:
            return self.save()
    @property
    def prog_info(self):
        """
            include epochs
        """
        meta = deepcopy(self.model_conf)
        meta['epochs'] = self.epochs
        return meta

    @property
    def trained(self):
        return True if self.epochs > 0 else False

    def get_batch(self, data, label, i, data_file=None, batch_size = None):
        if not batch_size : batch_size = self.batch_size
        last_batch = False if i < self.no_batch else True
        batch_offset = self.batch_size * i
        batch_size = self.batch_size if not last_batch else len(label)-batch_offset
        if data is not None:
            batch_x = data[batch_offset:batch_offset+batch_size]
        else:
            batch_file = data_file[batch_offset:batch_offset+batch_size]
            batch_x = np.array([image_load(data_file) for data_file in batch_file]) # TODO : crop

        if label is not None:
            batch_y = label[self.batch_size*i:self.batch_size*(i+1)] if not last_batch else label[self.batch_size*i:]
        else:
            batch_y = None

        return batch_x, batch_y
