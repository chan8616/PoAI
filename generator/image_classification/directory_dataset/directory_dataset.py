from argparse import Namespace
from collections import OrderedDict
from typing import Type
from gooey import Gooey, GooeyParser

from keras.preprocessing.image import ImageDataGenerator  # type: ignore

from ..image_classification_generator_config import (
        ImageClassificationGeneratorConfig,
        image_classification_generator_config_parser,
        )
from .directory_dataset_config import (
        DirectoryDatasetConfig,
        directory_dataset_config_parser,
        #  directory_dataset_config,
        )
from .dataset_config_samples import (DIR_CIFAR10,
                                     DIR_MNIST,
                                     )

"""
def directory_dataset_parser(
        parser: GooeyParser = GooeyParser(),
        title="Generator",
        directory_dataset_config=DirectoryDatasetConfig(),
        image_classification_generator_config=(
                ImageClassificationGeneratorConfig()),
        directory_dataset_configs=OrderedDict([
            ('directory_cifar10', DIR_CIFAR10()),
            #  ('directory_cifar100', DGC_CIFAR100()),
            #  ('directory_mnist', DGC_MNIST()),
            #  ('directory_fashion_mnist', DGC_FashionMNIST()),
            ]),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    for k, v in directory_dataset_configs.items():
        directory_dataset_parser = subs.add_parser(k)
        directory_dataset_config_parser(
                directory_dataset_parser, k, v)
        image_classification_generator_config_parser(
                directory_dataset_parser, v)

    directory_parser = subs.add_parser('directory')
    directory_dataset_config_parser(
            directory_parser, directory_dataset_config)
    image_classification_generator_config_parser(
            directory_dataset_parser,
            image_classification_generator_config)

    return parser
"""


def directory_dataset(
        cmd: str,
        args: Namespace,
        ) -> DirectoryDatasetConfig:
    #  from typing import Any
    #  config: Any
    if 'cifar10' in cmd:
        config = DIR_CIFAR10()  # type: ignore
    elif 'mnist' in cmd:
        config = DIR_MNIST()  # type: ignore
    else:
        config = DirectoryDatasetConfig()  # type: ignore
    config.update(args)

    if hasattr(config, 'auto_download'):
        if config.auto_download:
            config.auto_download()

    return config

    """
    ICGConfig = image_classification_generator_config(directory_generator_args)
    DGConfig = directory_generator_config(directory_generator_args)

    class Config(ICGConfig, DGConfig):
        pass
    config = Config()
    config.display()

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
    """
