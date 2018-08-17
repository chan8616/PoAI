import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath('..')) # refer to sibling folder

from .ops import *
from utils.data import *
from utils.util import *




open_data = {'mnist':call_mnist}

def load_model(sess,               # tf.Session()
               dataset = None,     # dataset name, open data e.g. mnist or hand-crafted dataset
               classes = None,     #
               image_size = None,  # if it is given, image can be cropped or scaling
               train_data = None,  # folder with images
               test_data = None,   # folder with images
               train_label = None, # txt file
               test_label = None,  # (optional) txt file
               learning_rate = 1e-4,
               optimizer = 'gradient',
               beta1 = 0.5,   # (optional) for some optimizer
               beta2 = 0.99,  # (optional) for some optimizer
               batch_size = 64,
               epochs = 20,
               checkpoint_dir = "checkpoint",
               checkpoint_name = None,
               train = False,
               test = False,
               epoch_interval = None,
               step_interval = None
               ):

    if not train and not test: return
    net = LOGISTIC(sess, dataset, classes, image_size, learning_rate, optimizer, beta1, beta2, batch_size, epochs)
    load, epoch = net.load(checkpoint_dir, checkpoint_name)
    if train: net.train(epoch, checkpoint_dir, checkpoint_name, epoch_interval, step_interval, train_data, train_label)
    if test:
        aassert(train or load, " [@] Train model first.")
        net.test(test_data, test_label)



class LOGISTIC(object):
    """

    Scenario 1 : Using open data.
     In that case, following arguments are useless
        : classes, image_size, train_data, test_data, train_label, test_label

    Scenario 2 : Hand-crafted dataset.
     In that case, following arguments are useless.
        : dataset

    """
    def __init__(self,
                 sess,               # tf.Session()
                 dataset,     # dataset name, open data e.g. mnist or hand-crafted dataset
                 classes,     #
                 image_size,  # if it is given, image can be cropped or scaling
                 learning_rate = 1e-4,
                 optimizer = 'gradient',
                 beta1 = 0.5,   # (optional) for some optimizer
                 beta2 = 0.99,  # (optional) for some optimizer
                 batch_size = 64,
                 epochs = 20
                 ):

        self.sess = sess
        self.classes = classes
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.build_model()



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

    def build_model(self):

        self.X = tf.placeholder(tf.float32, [None]+self.image_size, name="input")
        self.Y = tf.placeholder(tf.float32, [None]+[self.classes], name='label')

        self.y_logit, self.y_pred = self.classifier()

        self.loss = cross_entropy(self.Y, self.y_logit)
        self.acc = get_accuracy(self.Y, self.y_logit)
        self.prediction = tf.argmax(self.y_pred, axis=1)

        if self.optimizer == 'gradient':
            self.optim = OPTIMIZER[self.optimizer](self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'adam':
            self.optim = OPTIMIZER[self.optimizer](self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.loss)
        else:
            raise NotImplementedError('.')

        self.saver = tf.train.Saver()


    """
    #1 By dictionary
        Train Data path : [Project Folder]/dataset/[data name]/Data/[Class name]/[Data_name].[format]
        Test Data path : [Project Folder]/dataset/[data name]/Data/test/[Data_name].[format]
    #2 By txt
        Data path : [Project Folder]/dataset/[data name]/Data/[Data_name].[format]
    Checkpoint directory : [Project Folder]/checkpoint/[Model_name]/[Epoch or Iteration].ckpt

    """
    def save(self, checkpoint_dir, epoch, name=None):

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        file_name = 'model.ckpt'

        self.saver.save(self.sess, os.path.join(checkpoint_dir,file_name), global_step=epoch)

    def load(self, checkpoint_dir, name=None):
        import re
        print(" [*] Reading checkpoints....")
        self.model_name = name if name else self._model_name
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        if os.path.exists(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                print(" [!] checkpoint name is {}".format(ckpt_name))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0)) if not name else 0 # TODO
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter+1
            else:
                print(" [@] Failed to read the checkpoint.")
                aassert(False)
        else:
            print(" [@] Failed to find the checkpoint.")
            self.sess.run(tf.global_variables_initializer())
            return False, 0

    def classifier(self, reuse=False):
        im_dim = np.product(np.array(self.image_size))
        with tf.variable_scope('classifier') as scope:
            if reuse:
                scope.reuse_variables()
            X = self.X if len(self.X.get_shape().as_list())==2 else tf.reshape(self.X, [-1, im_dim])
            h = linear(X, self.classes, 'linear') # TODO : multi-classes
        return h, tf.nn.softmax(h)

    def train(self, epoch_trained, checkpoint_dir, checkpoint_name, epoch_interval=None, step_interval=None,
              train_data = None, train_label = None):
        print(" [*] Train data preprocessing...")
        if self.dataset in open_data.keys():
            self.train_data, self.train_label, self.test_data, self.test_label = open_data[self.dataset]()
            real_shape = list(self.train_data.shape[1:])
            aassert(self.image_size == real_shape, "unmatch {} vs {}".format(self.image_size, real_shape)) # TODO : cropping
            if len(self.image_size) == 2:
                self.image_size += [1]
                self.train_data = self.train_data[:,:,:,np.newaxis]
                self.test_data = self.test_data[:,:,:,np.newaxis]
            aassert(self.classes == self.train_label.shape[-1], " [!] Invaild classes")
            self.N = self.train_label.shape[0]
            self.train_data_path, self.test_data_path = None, None
        else:
            # print(type(train_data), type(train_label))
            self.train_data, self.train_data_path, self.train_label, real_shape = data_input(train_data,True, train_label)
            # print(self.train_label)
            aassert(self.image_size == real_shape, "unmatch {} vs {}".format(self.image_size, real_shape)) # TODO : cropping
            aassert(self.classes == int(self.train_label.shape[-1]),
                     " [!] Invalid classes {} vs {}, {}".format(type(self.classes), type(self.train_label), self.train_label.shape[-1]))
            self.N = self.train_data.shape[0] if self.train_data else len(self.train_data_path)
        print(" [!] Train data preprocessing... is done.")

        self.no_batch = int(np.ceil(self.N/self.batch_size))

        global_step = 0
        # print(self.epochs, epoch_trained)
        if self.epochs <= epoch_trained:
            print(" [!] Training is already done.")
            return
        print(" [*] Training start...")
        plt.ion()
        fig=plt.figure('{}_training '.format(self.model_name))
        for epoch in range(self.epochs-epoch_trained):
            for step in range(self.no_batch):
                batch_x, batch_y = self.get_batch(self.train_data, self.train_label, step, self.train_data_path)
                global_step += 1
                feed_dict= {self.X:batch_x, self.Y:batch_y}
                self.sess.run(self.optim, feed_dict=feed_dict)
                if step_interval and np.mod(global_step, step_interval) == 0:
                    loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    print("[{}] iter, loss : {}".format(global_step, loss))
                    report_plot(loss, global_step-step_interval, self.model_name)
            if epoch_interval and np.mod(epoch, epoch_interval)==0:
                loss = self.sess.run(self.loss, feed_dict=feed_dict)
                report_plot(loss, epoch, self.model_name)
                print("[{}] epoch, loss : {}".format(epoch, loss))
            self.save(checkpoint_dir, epoch, checkpoint_name)
        print(" [!] Trainining is done.")


    def test(self, test_data = None, test_label=None):
        """
            Result
            - file :
             [sample name or order] [True class(if labels are given)] [prediction] [is it correct?(if given)]
             ...
             [sample name or order] [True class(if labels are given)] [prediction] [is it correct?(if given)]
             The number of samples : [N], Accuracy : [acc(if labels are given)] # at last
        """
        if self.dataset not in open_data.keys():
            self.test_data, self.test_data_path, self.test_label, image_size = data_input(test_data,False, test_label)
        N = self.test_data.shape[0] if self.test_data is not None else len(self.test_data_path)
        # print(len(self.test_label), N)

        print(" [*] Test start...")
        with open(self.model_name, 'w') as f:
            correct = 0.
            for ith in range(N):
                w = [ith]
                if self.test_data is not None:
                    pred = self.sess.run(self.prediction, feed_dict={self.X:self.test_data[ith][np.newaxis]}) # TODO :multi-label
                elif self.test_data_path is not None:
                    image = image_load(self.test_data_path[ith])
                    pred = self.sess.run(self.prediction, feed_dict={self.X:image[np.newaxis]})
                if self.test_label is not None:
                    # print(ith)
                    w.append(int(np.argmax(self.test_label[ith])))
                w.append(pred) # TODO : multi-label
                if self.test_label is not None:
                    acc = bool(pred == int(np.argmax(self.test_label[ith])))
                    w.append(acc)
                    correct += float(acc)
                for elem in w:
                    f.write('{} '.format(elem))
                f.write('\n')
            result = "The number of samples : [{}]".format(N)
            if self.test_label is not None:
                result += ", Accuracy : [{}]".format(correct/float(N))
            f.write(result)
        print(" [!] Test is done.")

    @property
    def _model_name(self):
        if not self.dataset:
            self.dataset = self.file_path[0].split('/')[-4]
        return '{}_{}_{}'.format('simple', 'logistic', self.dataset)
