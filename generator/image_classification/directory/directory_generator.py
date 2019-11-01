from collections import OrderedDict
from typing import Type
from gooey import Gooey, GooeyParser

from keras.preprocessing.image import ImageDataGenerator  # type: ignore

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

    directory_parser = subs.add_parser('directory')
    directory_generator_config_parser(
            directory_parser, directory_generator_config)

    return parser


def directory_generator(Generator: Type[ImageDataGenerator],
                        directory_generator_config: DirectoryGeneratorConfig):
    generators = [(None
                   if directory is '' else
                   Generator(rescale=1./255.).flow_from_directory(
                       directory,
                       target_size=directory_generator_config.TARGET_SIZE,
                       color_mode=directory_generator_config.COLOR_MODE,
                       class_mode=directory_generator_config.CLASS_MODE,
                       batch_size=directory_generator_config.BATCH_SIZE,
                       shuffle=directory_generator_config.SHUFFLE,
                       ))
                  for directory in [directory_generator_config.DIRECTORY,
                                    directory_generator_config.VAL_DIRECTORY]]

    return generators
