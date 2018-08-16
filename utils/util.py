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

class Redirection(object):
    def __init__(self, log_area):
        self.out = log_area

    def write(self, string):
        self.out.WriteText(string+'\n')

def shuffle(x, y):
    from random import shuffle as sf
    d = list(zip(x,y))
    sf(d)
    x_, y_ = zip(*d)
    return np.array(x_).astype(float), np.array(y_).astype(float)


def pickle_save(d, file_name):
    with open("{}.pickle".format(file_name), 'wb') as f:
        pickle.dump(d, f)

def pickle_load(file_name):
    with open("{}.pickle".format(file_name), 'rb') as f:
        return pickle.load(f, encoding='bytes')

def image_load(file_path):            # load image as float numpy array
    return np.array(Image.open(file_path)).astype(float)/255.

def aassert(statement, message=''): #TODO
    if not statement:
        pprint(message)
        assert False, message

def report_plot(data, i, model_name, log='./log'):
    print(data,i)
    if not os.path.exists(log):
        os.mkdir(log)
    if i==0 or not os.path.exists(os.path.join(log,"{}.pickle".format(model_name))):
        with open(os.path.join(log,"{}.pickle".format(model_name)), 'w'):
            pass
        pickle_save([[data], [i]], os.path.join(log,model_name))
        return
    d, t = pickle_load(os.path.join(log,model_name))
    d.append(data)
    t.append(i)
    pickle_save([d,t], os.path.join(log,model_name))
    plt.plot(t,d)
    plt.pause(0.0001)
