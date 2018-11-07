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

from sklearn.metrics import confusion_matrix
# import pandas as pd
# import seaborn as sns

def conf_mtx(y_true, y_pred, label_names=None):
    mtx = confusion_matrix(y_true, y_pred)
    if label_names is None or len(mtx) != len(label_names):
        label_names = list(np.arange(len(y_true)))
    print(mtx)
    # cm = pd.DataFrame(mtx,columns=label_names,index=label_names)
    # try:
    # heatmap = sns.heatmap(cm, annot=True, fmt='d')
    # except:
    #     raise ValueError("Confusion matrix values must be integers.")
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
                 input_shape,
                 checkpoint_name = 'model',
                 additional_layer = [1024]
                 ):

        self.add_layer= additional_layer
        # 1. Define a name and a checkpoint path of the model

        self.model_name = name
        checkpoint_dir = path.join(checkpoint_dir, model, dataset_name)
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
                               'input_shape':input_shape, #
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

    def test(self, x, y=None, label_name=None, visualize=False):
        y_pred_score = self.predict(x)
        y_pred = np.argmax(y_pred_score, axis=1)
        from math import log10
        idx = int(log10(len(y_pred)))+1
        classes = int(log10(y_pred_score.shape[1]))+1
        model_result = './log'
        if y is not None:
            if x.shape[0] == 1: # A image
                print("The prediction is [{}], result : [{}]".format(np.argmax(y_pred), np.argmax(y_pred)==y))
            else:
                if not path.exists(model_result):
                    makedirs(model_result)
                with open(path.join(model_result, self.model_name+'_eval.txt'), 'w') as f:
                    f.write('{} | {} | {} | {}\n'.format('order'.rjust(idx),
                                                       'pred'.rjust(classes),
                                                       'label'.rjust(classes),
                                                       'Correct'))
                    f.write('-' * (max(idx,5) + 3*3+7 + max(classes,4)+max(classes,5))+'\n')
                    for i,v in enumerate(y_pred):
                        cor = 'True' if v == y[i] else 'False'
                        f.write('{} | {} | {} | {}\n'.format(str(i).zfill(idx).rjust(5),
                                                           str(v).zfill(classes).rjust(4),
                                                           str(y[i]).zfill(classes).rjust(5),
                                                           cor.rjust(7)))
                    f.write('The number of samples : [{}], Accuracy : [{:.4f}]'.format(y_pred.shape[0], self.accuracy(x,y)))
            print('The number of samples : [{}], Accuracy : [{:.4f}]'.format(y_pred.shape[0], self.accuracy(x,y)))
            if visualize:
                y = np.squeeze(y)
                y_pred = np.argmax(self.predict(x), axis=1)
                if len(y.shape) > 1: #
                    y = np.argmax(y, axis=1)
                conf_mtx(y, y_pred, label_name)
        else:
            if x.shape[0] == 1: # A image
                print("The result is [{}]".format(np.argmax(y_pred)))
            else:
                if not path.exists(model_result):
                    makedirs(model_result)
                with open(path.join(model_result, self.model_name+'_predict.txt'), 'w') as f:
                    f.write('{} | {}\n'.format('order'.rjust(idx), 'pred'.rjust(classes)))
                    f.write('-' * (max(idx,5) + 3 + max(classes,4))+'\n')
                    for i,v in enumerate(y_pred):
                        f.write('{} | {}\n'.format(str(i).zfill(idx).rjust(5), str(v).zfill(classes).rjust(4)))
                    f.write('The number of samples : [{}]'.format(y_pred.shape[0]))
            print('The number of samples : [{}]'.format(y_pred.shape[0]))

    def test_with_generator(self, generator, steps, label_name=None, visualize=False):
        y_pred_score = self.predict_with_generator(generator, steps)
        y_pred = np.argmax(y_pred_score, axis=1)

    def merge_callbacks(self, conf):
        name, batch_size = self.model_conf['name'], self.model_conf['batch_size']

        print(conf['ntrain'], batch_size)
        print(int(np.ceil(conf['ntrain']/batch_size)))
        print(int(np.ceil(conf['ntrain']/batch_size)*conf['step_interval']))

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
    def predict_with_generator(self, generator, steps):
        return self.model.predict_generator(generator=generator(with_y=False),
                                            steps=steps)
    def accuracy(self, x, y):
        assert y is not None
        y_pred = self.predict(x)
        return np.mean(np.equal(np.argmax(y_pred, axis=1), y).astype(np.float))

    def accuracy_with_generator(self, generator, steps):
        return self.model.evaluate_generator(generator=generator(),
                                             steps=steps)['acc']
    def train_with_generator(self,
                            generator,
                            valid_generator,
                            steps,
                            valid_steps,
                            epochs,
                            period,
                            num_x,
                            step_interval,
                            save=True):

        debug_conf = {'log_dir':path.join('./logs', self.model_name),
                      'ntrain' : num_x,
                      'save_best_only':True,
                      'ckpt_path':self.model_ckpt,
                      'step_interval':step_interval,
                      'period':period,
                      'patience':2}
        if self.epochs >= epochs:
            print("[!] Already Done.")
            return
        self.model.fit_generator(generator=generator(),
                                 steps_per_epoch=steps,
                                 epochs=epochs,
                                 verbose=0,
                                 callbacks=self.merge_callbacks(debug_conf),
                                 validation_data = valid_generator(),
                                 validation_steps = valid_steps,
                                 shuffle=True,
                                 initial_epoch = self.epochs)
        self.epochs = epochs
        if save:
            self.save()

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

        if self.epochs >= epochs:
            print("[!] Already Done.")
            return
        self.model.fit(x=x,
                       y=one_hot(y, classes=self.num_classes),
                       epochs=epochs,
                       validation_split=0.01,
                       initial_epoch=self.epochs,
                       callbacks=self.merge_callbacks(debug_conf),
                       verbose=0)
        self.epochs = epochs
        if save:
            self.save()
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
