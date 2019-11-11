from queue import Queue

import tensorflow as tf
from keras.callbacks import Callback


class KerasQueueLogger(Callback):
    def __init__(self, stream: Queue, count_mode='steps'):
        super(KerasQueueLogger, self).__init__()
        self.stream = stream
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        if self.use_steps:
            self.target = self.params['steps']
        else:
            self.target = self.params['samples']

        self.stream.put((f'Epoch {epoch}', None, None))

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get('num_steps', 1)
        loss = logs.get('loss', None)
        acc = logs.get('acc', None)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        self.stream.put(('batch', (self.seen, self.target, loss, acc), None))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', None)
        acc = logs.get('acc', None)
        val_loss = logs.get('val_loss', None)
        val_acc = logs.get('val_acc', None)
        current_epoch = epoch + 1

        self.stream.put(('epoch', (current_epoch, loss, acc, val_loss, val_acc), None))

    def on_train_end(self, logs=None):
        self.stream.put('end')


def tf_logger(stream: Queue):
    def put_in_stream(tensors):
        _loss = tensors.get('loss', None)
        _acc = tensors.get('accuracy', None)
        stream.put(('batch', (0, 1, _loss, _acc), None))
    loss, accuracy = 0., 0.
    log_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": accuracy},
                                          every_n_iter=10, at_end=True,
                                          formatter=put_in_stream)
    return log_hook
