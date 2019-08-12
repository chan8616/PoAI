import wx

from keras.callbacks import Callback
from utils import MyProgbar


class MyProgbarLogger(Callback):
    """Callback that prints metrics to stdout.
    Arguments:
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
    Raises:
            ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, wxObject, count_mode='samples', stateful_metrics=None):
        super(MyProgbarLogger, self).__init__()
        self.wxObject = wxObject
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        self.stateful_metrics = set(stateful_metrics or [])

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
        self.progbar = MyProgbar(
                wxObject=self.wxObject,
                target=self.target,
                verbose=self.verbose,
                stateful_metrics=self.stateful_metrics,
                unit_name='step' if self.use_steps else 'sample')

    def on_batch_begin(self, batch, logs=None):
        self.log_values = []

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

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and (self.target is None or self.seen < self.target):
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)
