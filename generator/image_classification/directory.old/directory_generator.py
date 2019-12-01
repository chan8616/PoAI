from argparse import Namespace
from collections import OrderedDict
from typing import Type
from gooey import Gooey, GooeyParser

from keras.preprocessing.image import ImageDataGenerator  # type: ignore

from ..image_classification_generator_config import (
        image_classification_generator_config,
        image_classification_generator_config_parser,
        )
from .directory_generator_config import (
        DirectoryGeneratorConfig,
        directory_generator_config_parser,
        directory_generator_config,
        )
from .config_samples import (DGC_CIFAR10,
                             DGC_CIFAR100,
                             DGC_MNIST,
                             DGC_FashionMNIST,
                             )


def directory_generator_parser(
        parser: GooeyParser = GooeyParser(),
        title="Generator",
        directory_generator_config=DirectoryGeneratorConfig(),
        directory_generator_configs=OrderedDict([
            ('directory_cifar10', DGC_CIFAR10()),
            ('directory_cifar100', DGC_CIFAR100()),
            ('directory_mnist', DGC_MNIST()),
            ('directory_fashion_mnist', DGC_FashionMNIST()),
            ]),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    for k, v in directory_generator_configs.items():
        directory_generator_parser = subs.add_parser(k)
        directory_generator_config_parser(
                directory_generator_parser, k, v)
        image_classification_generator_config_parser(
                directory_generator_parser, v)

    directory_parser = subs.add_parser('directory')
    directory_generator_config_parser(
            directory_parser, directory_generator_config)
    image_classification_generator_config_parser(
            directory_generator_parser, v)

    return parser


def directory_generator(generator: ImageDataGenerator,
                        directory_generator_args: Namespace,
                        ):
    ICGConfig = image_classification_generator_config(directory_generator_args)
    DGConfig = directory_generator_config(directory_generator_args)

    class Config(ICGConfig, DGConfig):  # type: ignore
        pass
    config = Config()

    if hasattr(directory_generator_args, 'auto_download'):
        if directory_generator_args.auto_download:
            config.auto_download()

    generators = [generator.flow_from_directory(
                       directory=config.DIRECTORY,
                       target_size=config.TARGET_SIZE,
                       color_mode=config.COLOR_MODE,
                       class_mode=config.CLASS_MODE,
                       batch_size=config.BATCH_SIZE,
                       shuffle=True,
                       ),
                  (None
                   if config.VAL_DIRECTORY is '' else
                   generator.flow_from_directory(
                       directory=config.VAL_DIRECTORY,
                       target_size=config.TARGET_SIZE,
                       color_mode=config.COLOR_MODE,
                       class_mode=config.CLASS_MODE,
                       batch_size=config.BATCH_SIZE,
                       shuffle=False,
                       ))]

    return generators
