import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append(os.path.abspath('..')) # refer to sibling folder

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
                 sess,
                 model_name,
                 dataset,
                 learning_rate,
                 optimizer,
                 beta1,
                 beta2,
                 batch_size,
                 epochs,
                 model_dir):

        self.sess = sess
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.build_model()

    def build_optimzier(self, loss, var_list=None):
        assert self.optimizer in ['gradient', 'adam']
        import tensorflow.train as opt

        var_list = tf.trainable_variables() if var_list is None else var_list

        if self.optimizer == 'gradient':
            self.optim = opt.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        elif self.optimizer == 'adam':
            self.optim = opt.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(loss)
        else:
            raise NotImplementedError('.')

    def build_classification(self, classes):
        self.classes = classes

    def build_model(self):
        raise NotImplementedError('')

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
        dataset_name = self.dataset if self.dataset is not None else self.file_path[0].split('/')[-4]
        return '{}_{}'.format(dataset_name, self.model_name)
