from collections import OrderedDict
from typing import Type
from gooey import Gooey, GooeyParser

from keras.preprocessing.image import ImageDataGenerator  # type: ignore

from .dataframe_generator_config import (
        ImageClassificationGeneratorConfig,
        DataframeGeneratorConfig,
        dataframe_generator_config_parser,
        dataframe_generator_config,
        )
#  from .config_samples import (DGC_CIFAR10,
#                               DGC_CIFAR100,
#                               DGC_MNIST,
#                               DGC_FashionMNIST,
#                               )


def dataframe_generator_parser(
        parser: GooeyParser = GooeyParser(),
        title="Generator",
        dataframe_generator_config=DataframeGeneratorConfig(),
        dataframe_generator_configs=OrderedDict([
            #  ('dataframe_cifar10', DGC_CIFAR10()),
            #  ('dataframe_cifar100', DGC_CIFAR100()),
            #  ('dataframe_mnist', DGC_MNIST()),
            #  ('dataframe_fashion_mnist', DGC_FashionMNIST()),
            ]),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    for k, v in dataframe_generator_configs.items():
        dataframe_generator_parser = subs.add_parser(k)
        dataframe_generator_config_parser(
                dataframe_generator_parser, k, v)

    dataframe_parser = subs.add_parser('dataframe')
    dataframe_generator_config_parser(
            dataframe_parser, dataframe_generator_config)

    return parser


def dataframe_generator(generator: ImageDataGenerator,
                        config: ImageClassificationGeneratorConfig,
                        dataframe_generator_config: DataframeGeneratorConfig):
    generators = [generator.flow_from_dataframe(
                       dataframe=dataframe_generator_config.DATAFRAME,
                       directory=dataframe_generator_config.DIRECTORY,
                       target_size=config.TARGET_SIZE,
                       color_mode=config.COLOR_MODE,
                       class_mode=config.CLASS_MODE,
                       batch_size=config.BATCH_SIZE,
                       shuffle=True,
                       ),
                  (None
                   if dataframe_generator_config.VAL_DATAFRAME is '' else
                   generator.flow_from_dataframe(
                       dataframe=dataframe_generator_config.VAL_DATAFRAME,
                       directory=dataframe_generator_config.VAL_DIRECTORY,
                       target_size=config.TARGET_SIZE,
                       color_mode=config.COLOR_MODE,
                       class_mode=config.CLASS_MODE,
                       batch_size=config.BATCH_SIZE,
                       shuffle=False,
                       ))]

    return generators
