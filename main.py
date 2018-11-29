import os
import numpy as np

import tensorflow as tf
import time

from utils.util import image_load
from run import load_model, data_select, select_optimizer, get_data_list

flags = tf.app.flags

flags.DEFINE_boolean('train', False, "")
flags.DEFINE_string('dataset', 'cifar10' , "")
flags.DEFINE_float('learning_rate', 0.001, "")
flags.DEFINE_integer('batch_size', 64, "")
flags.DEFINE_string('optimizer', 'gradient', "")
flags.DEFINE_float('step_interval', 0.1, "")
flags.DEFINE_integer('max_epochs', 100, "")
flags.DEFINE_string('model', 'logistic', "")
flags.DEFINE_integer('gpu', 0, "")
flags.DEFINE_string('name', 'ver1', '')

FLAGS = flags.FLAGS

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

# dataset

open_data_list = get_data_list()

#TODO : generalize to non-image data
#TODO : implement training
if not FLAGS.dataset in open_data_list:
    path = os.path.join('.', 'Dataset', FLAGS.dataset)
    if not os.path.exists(path): # exists
        if os.path.isfile(path): # file
            fname, ext = os.path.splitext(path)
            assert ext in ['png', 'jpg', 'jpeg']
            data = np.expand_dims(image_load(path), 0) # (1, h, w, c)
        elif os.path.isdir(path): # directory
            flist = os.listdir()
            imgs_path = [f for f in flist if f.split('.')[-1] in ['png, jpg, jpeg']]
            assert len(imgs_path) > 0, "There is no data in the directory"
            data = np.array([image_load(img_path) for img_path in imgs_path]).astype(np.float) # (n, h, w, c)
    else:
        assert False, "There is no such data or directory."
    dataspec = {'name' : FLAGS.dataset,
                'test_x' : data,
                'label_names':None,
                'input_shape':(224, 224, 3),
                'data_type':'I'} #TODO
else: # use open data
    dataspec = {'name' : FLAGS.dataset,
                'label_names':None,
                'input_shape':(224, 224, 3),
                'data_type':'I'} #TODO

data, provider = data_select(dataspec, FLAGS.train)
opt = select_optimizer(FLAGS.optimizer, FLAGS.learning_rate) if FLAGS.train else None

load_model(model_name = FLAGS.model,
           name = FLAGS.name,
           dataset_name = FLAGS.dataset,
           data=data,
           data_provider=provider,
           optimizer=opt,
           epochs=FLAGS.max_epochs,
           batch_size = FLAGS.max_epochs,
           train=FLAGS.train,
           step_interval=FLAGS.step_interval)