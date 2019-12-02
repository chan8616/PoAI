import os
from pathlib import Path

import pandas as pd

from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.layers import GlobalAveragePooling2D  # type: ignore
from keras.layers import GlobalMaxPooling2D  # type: ignore
from keras.callbacks import ModelCheckpoint  # type: ignore
from keras.optimizers import sgd, adam

from model.model_config import ModelConfig
from .build_config import LinearBuildConfig as BuildConfig
from model.keras_applications import model as modellib


def log(text, array=None):
    pass


class LinearModel(modellib.KerasAppBaseModel):
    def __init__(self, model_config=ModelConfig()):
        #  super(Model, self).__init__(Model, model_config)
        self.model_config = model_config
        self.model_dir = model_config.MODEL_DIR

    def build(self, build_config=BuildConfig()
              ) -> None:
        self.build_config = build_config

        inp = Input(build_config.FLATTEN_INPUT_SHAPE)
        if len(inp.shape) > 2:
            inp = Flatten()(inp)
        x = inp

        for i, nodes in enumerate(build_config.HIDDEN_LAYERS):
            x = Dense(nodes, activation='relu',
                      name='fc{}'.format(i))(x)
        x = Dense(build_config.TARGET_SIZE)(x)

        self.keras_model = Model(
            inputs=inp, 
            outputs=x,
            name=build_config.NAME)

        self.set_log_dir()

    def train(self, train_config, train_generator, val_generator,
              custom_callbacks=None):
        """Train the model.
        train_config:
        train_generator, val_generator: Training and validation Dataset objects.
        custom_callbacks: 
        """

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        if val_generator is not None:
            train_config.MONITOR = 'val_' + train_config.MONITOR
        callbacks = [ModelCheckpoint(self.checkpoint_path,
                                     monitor=train_config.MONITOR,
                                     verbose=0, save_weights_only=True)]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(
            self.epoch, train_config.LEARNING_RATE))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        # self.TRAIN_LAYERS = train_config.TRAIN_LAYERS
        self.set_trainable(train_config.FREEZE_LAYER)
        #  self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        if train_config.OPTIMIZER == 'sgd':
            optimizer = sgd(lr=train_config.LEARNING_RATE)
        elif train_config.OPTIMIZER == 'adam':
            optimizer = adam(lr=train_config.LEARNING_RATE)
        else:
            raise AttributeError(f'{train_config.OPTIMIZER} is not exist!')

        self.keras_model.compile(optimizer=optimizer,
                                 loss=train_config.LOSS
                                 #  metrics=['accuracy'],
                                 )

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        #  if os.name is 'nt':
        #      workers = 0
        #  else:
        #      workers = multiprocessing.cpu_count()
        if val_generator is not None:
            self.keras_model.fit_generator(
                train_generator,
                initial_epoch=self.epoch,
                epochs=train_config.EPOCHS,
                steps_per_epoch=len(train_generator),
                callbacks=callbacks,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                #  max_queue_size=100,
                #  workers=workers,
                #  use_multiprocessing=True,
            )
        else:
            self.keras_model.fit_generator(
                train_generator,
                initial_epoch=self.epoch,
                epochs=train_config.EPOCHS,
                steps_per_epoch=len(train_generator),
                callbacks=callbacks,
                #  max_queue_size=100,
                #  workers=workers,
                #  use_multiprocessing=True,
            )
        self.epoch = max(self.epoch, train_config.EPOCHS)

    def test(self, test_generator, result_save_path, stream=None):
        #  now = datetime.datetime.now()
        #  result_dir = Path("{}{:%Y%m%dT%H%M}".format(
        #          str(Path(test_args.result_path).parent), now))
        result_dir = Path(result_save_path).parent
        if not result_dir.exists():
            result_dir.mkdir(parents=True)

        columns = ['prediction']

        df = pd.DataFrame(columns=columns)
        df.to_csv(result_save_path, index=False)
        steps_done = 0
        total = len(test_generator)

        if stream is not None:
            stream.put(('Testing...', None, None))

        while steps_done < total:
            idx = next(test_generator.index_generator)
            x, y = test_generator._get_batches_of_samples(idx)

            pred = self.keras_model.predict_on_batch(x)
            print(x.shape, pred.shape)

            df = pd.DataFrame(
                    pred,
                    columns=columns)
            df.to_csv(result_save_path, mode='a', index=False, header=False)

            steps_done += 1
            if stream is not None:
                stream.put(('test', (steps_done, total), None))

        if stream is not None:
            stream.put('end')

    def set_trainable(self, layers):
        for layer in self.keras_model.layers[
                :layers]:
            layer.trainable = False
        for layer in self.keras_model.layers[
                layers:]:
            layer.trainable = True
