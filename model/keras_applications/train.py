#  from typing import Union, Callable
#  from argparse import Namespace
from gooey import GooeyParser

from .train_config import TrainConfig, train_config_parser  # , train_config
from ..utils.stream_callbacks import KerasQueueLogger
#  from .config_samples import (BalloonConfig, CocoConfig,
#                               NucleusConfig, ShapesConfig)
#  from ..utils.stream_callbacks import KerasQueueLogger

#  from keras.callbacks import ModelCheckpoint


def train_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        train_config=TrainConfig(),
        train_imagenet_config=TrainConfig(),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    train_parser = subs.add_parser('train')
    train_config_parser(train_parser, title, train_config)

    train_parser = subs.add_parser('train_imagenet')
    train_config_parser(train_parser, title, train_imagenet_config)

    return parser


def train(model, train_args, train_generator, val_generator, stream
          ) -> None:
    """Train the model."""
    callbacks = [KerasQueueLogger(stream)]
    train_config = TrainConfig()
    train_config.update(train_args)
    model.train(train_config,
                train_generator, val_generator,
                custom_callbacks=callbacks)
    """
    callbacks = [KerasQueueLogger(stream),
                 ModelCheckpoint(model.checkpoint_path,
                                 monitor=train_args.monitor,
                                 verbose=0, save_weights_only=True)]

    log("\nStarting at epoch {}. LR={}\n".format(
       train_args.epoch, train_args.learning_rate))
    log("Checkpoint Path: {}".format(self.checkpoint_path))
    #  self.set_trainable(layers)
    optimizer = train_args.optimizer
    model.compile(optimizer=optimzer,
                  loss='sparse_categorical_crossentropy')

    print("Training network heads")
    model.fit_generator(train_generator,
                        initial_epoch=model.epoch,
                        epochs=train_args.epochs,
                        callbacks=callbacks,
                        # validation_split=validation_split,
                        validation_data=validation_generator,
                        validation_steps=train_args.validation_steps,
                        #  shuffle=train_args.shuffle,
                        # initial_epoch=initial_epoch,
                        # steps_per_epoch=steps_per_epoch,
                        )

    #  model.train(dataset_train, dataset_val,
    #              learning_rate=train_args.learning_rate,
    #              epochs=train_args.epochs,
    #              layers='heads',
    #              custom_callbacks=[callback])
    """
