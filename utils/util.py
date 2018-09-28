"""
    utils.py
    Utility function for project
"""
import pickle
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import wx

from os import makedirs, path, stat
import sys
import urllib




def _download(url, directory, file_name=None):
    file_path = path.join(directory, file_name)
    if not path.exists(directory):
        makedirs(directory)
    if not path.isfile(file_path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {} {:.1f} %'.format(
                file_name, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        file_path, _ = urllib.request.urlretrieve(url, file_path, _progress)
        print()
        statinfo = stat(file_path)
        print('Successfully donloaded', file_name, statinfo.st_size, 'bytes.')
    return file_path

class Redirection(object):
    def __init__(self, log_area):
        self.out = log_area

    def write(self, string):
        self.out.WriteText(string)

    def flush(self):
        pass

def shuffle(x, y):
    from random import shuffle as sf
    d = list(zip(x,y))
    sf(d)
    x_, y_ = zip(*d)
    return np.array(x_).astype(x.dtype), np.array(y_).astype(y.dtype)


def pickle_save(d, file_name):
    #    with open("test.pickle", 'wb') as f:
    with open("{}.pickle".format(file_name), 'wb') as f:
        pickle.dump(d, f)

def pickle_load(file_name):
    with open("{}.pickle".format(file_name), 'rb') as f:
        return pickle.load(f, encoding='bytes')

def image_load(file_path, resize=0):            # load image as float numpy array
    img = Image.open(file_path).resize((resize,resize)) if resize > 0 else Image.open(file_path)
    return np.array(img).astype(float)/255.

def aassert(statement, message=''): #TODO
    if not statement:
        print(message)
        assert False, message

def report_plot(data, i, model_name, log='./log'):
    if not os.path.exists(log):
        os.mkdir(log)
    if i==0. or not os.path.exists(os.path.join(log,"{}.pickle".format(model_name))):
        pickle_save([[data], [i]], os.path.join(log,model_name))
        return
    d, t = pickle_load(os.path.join(log,model_name))
    d.append(data)
    t.append(i)
    pickle_save([d,t], os.path.join(log,model_name))
    plt.plot(t,d)
    plt.pause(0.0001)

def gpu_inspection():
    from tensorflow.python.client import device_lib
    device_info = device_lib.list_local_devices()
    cpu = [dev for dev in device_info if 'CPU' in dev.name][0]
    gpu = [dev for dev in device_info if 'GPU' in dev.name]
    print("The number of gpu is {}".format(len(gpu)))
    return len(gpu)

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, model_name, batch_size, ntrain, step_interval=0.1):
        self.ntrain = ntrain
        self.batch_size = batch_size
        self.step = np.ceil(ntrain/batch_size).astype(np.int64)
        self.model_name = model_name
        self.step_interval = int(self.step*step_interval)
    def on_train_begin(self, logs={}):
        print("[@] trainig start...")
    def on_train_end(self, logs={}):
        print("[@] trainig is done.")
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        print('[{}] epoch_end, val_loss : [{}], val_acc : [{}]' \
                    .format(epoch, val_loss, val_acc))
    def on_batch_end(self, batch, logs={}):
        if batch % self.step_interval > 0 or batch == self.step:
            return
        loss = logs.get('loss')
        acc = logs.get('acc')
        report_plot(loss, float(self.epoch)+batch/self.step, self.model_name)
        print('[{}] epoch [{}/{}], loss : [{:.4f}], acc : [{}]' \
                    .format(self.epoch, batch*self.batch_size, self.ntrain, loss, acc))
