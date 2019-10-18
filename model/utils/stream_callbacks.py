from queue import Queue

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

        if self.verbose:
            if self.epochs > 1:
                print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get('num_steps', 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        self.stream.put((self.seen, self.target, ""))

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.stream.put('end')