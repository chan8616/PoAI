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
        directory_generator_config=DirectoryGeneratorConfig(),
        cifar10_config=DGC_CIFAR10(),
        cifar100_config=DGC_CIFAR100(),
        mnist_config=DGC_MNIST(),
        fashion_mnist_config=DGC_FashionMNIST(),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    directory_parser = subs.add_parser('directory')
    directory_generator_config_parser(
            directory_parser, directory_generator_config)

    directory_parser = subs.add_parser('directory_cifar10')
    directory_generator_config_parser(
            directory_parser, cifar10_config, auto_download=True)

    directory_parser = subs.add_parser('directory_cifar100')
    directory_generator_config_parser(
            directory_parser, cifar100_config, auto_download=True)

    directory_parser = subs.add_parser('directory_mnist')
    directory_generator_config_parser(
            directory_parser, mnist_config, auto_download=True)

    directory_parser = subs.add_parser('directory_fashion_mnist')
    directory_generator_config_parser(
            directory_parser, fashion_mnist_config, auto_download=True)

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
