import os
import re
import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from keras.models import Model  # type: ignore
from keras.layers import Dense, Flatten  # type: ignore
from keras.layers import GlobalAveragePooling2D  # type: ignore
from keras.layers import GlobalMaxPooling2D  # type: ignore
from keras.callbacks import ModelCheckpoint  # type: ignore
from keras.optimizers import sgd, adam

from ..model_config import ModelConfig
from .build_config import BuildConfig


def log(text, array=None):
    pass


class KerasAppBaseModel():
    def __init__(self, BaseModel, model_config=ModelConfig()):
        self.BaseModel = BaseModel
        self.model_config = model_config
        self.model_dir = model_config.MODEL_DIR

    def build(self, build_config=BuildConfig()
              ) -> None:
        self.build_config = build_config

        keras_model = self.BaseModel(
                include_top=False,
                weights=None,
                input_shape=build_config.INPUT_SHAPE)

        x = keras_model.output
        if 'flatten' == build_config.POOLING:
            x = Flatten(name='flatten')(x)
        elif 'avg' == build_config.POOLING:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif 'max' == build_config.POOLING:
            x = GlobalMaxPooling2D(name='max_pool')(x)
        else:
            raise NotImplementedError('Wrong pooling')

        for i, nodes in enumerate(build_config.HIDDEN_LAYERS):
            x = Dense(nodes, activation='relu',
                      name='fc{}'.format(i))(x)
        x = Dense(build_config.CLASSES, activation='softmax')(x)

        self.keras_model = Model(
            inputs=keras_model.input,
            outputs=x,
            name=build_config.NAME)
        self.model_dir = build_config.LOG_DIR
        self.set_log_dir()

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.build_config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(
                    self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.build_config.NAME.lower()), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(
                    dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py  # type: ignore
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving  # type: ignore

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = (keras_model.inner_model.layers
                  if hasattr(keras_model, "inner_model")
                  else keras_model.layers)

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        pass

    def set_trainable(self, layers,  # keras_model=None,
                      indent=0, verbose=1):
        """
        layers: Allows selecting wich layers to train. It can be:
            - One of these predefined values:
              heads: Classifier heads of the network
              all: All the layers
              ...
              3+: Train Block 3 and up
              4+: Train Block 4 and up
              5+: Train Block 5 and up
              ...
        """
        pass

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})" \
                    r"[/\\]*[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(
                        int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based,
                #   and in Keras code it's 0-based.
                # So, adjust for that then increment by one
                #   to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.build_config.NAME.lower(), now))

        # Path to save after each epoch.
        # Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
                self.log_dir,
                "{}_*epoch*.h5".format(
                    self.build_config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_config, train_generator, val_generator,
              custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
                custom_callbacks: Optional. Add custom callbacks to be called
                with the keras fit_generator method.
                Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
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
        self.set_trainable(train_config.TRAIN_LAYERS[train_config.TRAIN_LAYER])
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
                verbose=0,
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
                verbose=0,
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

        filepaths = np.array(test_generator._filepaths)
        columns = ['filename', 'prediction']

        df = pd.DataFrame(columns=columns)
        df.to_csv(result_save_path, index=False)
        steps_done = 0
        total = len(test_generator)

        if stream is not None:
            stream.put(('Testing...', None, None))

        while steps_done < total:
            idx = next(test_generator.index_generator)
            x, y = test_generator._get_batches_of_transformed_samples(idx)

            pred = self.keras_model.predict_on_batch(x)

            df = pd.DataFrame(
                np.stack(
                    [filepaths[idx], np.argmax(pred, axis=1)],
                    axis=1),
                columns=columns)
            df.to_csv(result_save_path, mode='a', index=False, header=False)

            steps_done += 1
            if stream is not None:
                stream.put(('test', (steps_done, total), None))

        if stream is not None:
            stream.put('end')

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    """
    def get_trainable_layers(self):
    """
    #      """Returns a list of layers that have weights."""
    """
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
    """
