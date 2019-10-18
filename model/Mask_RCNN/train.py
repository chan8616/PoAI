#  from typing import Union, Callable
from argparse import Namespace
from gooey import Gooey, GooeyParser

from .train_config import train_config_parser, train_config
from .config_samples import (BalloonConfig, CocoConfig,
                             NucleusConfig, ShapesConfig)
from ..utils.stream_callbacks import KerasQueueLogger


def train_parser(
        parser: GooeyParser = GooeyParser(),
        title="train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    balloon_train_parser = subs.add_parser('train_balloon')
    train_config_parser(balloon_train_parser,
                        BalloonConfig(),)

    train_parser = subs.add_parser('train')
    train_config_parser(train_parser)

    #  balloon_train_parser = subs.add_parser('train_balloon')
    #  train_config_parser(balloon_train_parser,
    #                      BalloonConfig(),)

    #  coco_train_parser = subs.add_parser('train_coco')
    #  train_config_parser(coco_train_parser,
    #                      CocoConfig(),)

    #  nucleus_train_parser = subs.add_parser('train_nucleus')
    #  train_config_parser(nucleus_train_parser,
    #                      NucleusConfig(),)

    #  shapes_train_parser = subs.add_parser('train_shapes')
    #  train_config_parser(shapes_train_parser,
    #                      ShapesConfig(),)

    return parser
    #  model = compile_.compile_(args)
    #  return (model, args.epochs,
    #          args.epochs if args.validation_steps is None
    #          else args.validation_steps,
    #          get_callbacks(args), args.shuffle)


def train(t_config):
    """Train the model."""
    # Training dataset.
    #  dataset_train = BalloonDataset()
    #  dataset_train.load_balloon(args.dataset, "train")
    #  dataset_train.prepare()

    # Validation dataset
    #  dataset_val = BalloonDataset()
    #  dataset_val.load_balloon(args.dataset, "val")
    #  dataset_val.prepare()
    model, train_args, dataset_train, dataset_val, stream = t_config
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    callback = KerasQueueLogger(stream)
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=train_args.learning_rate,
                epochs=train_args.epochs,
                layers='heads',
                custom_callbacks=[callback])
