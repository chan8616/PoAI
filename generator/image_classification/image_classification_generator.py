from argparse import Namespace
from collections import OrderedDict
from typing import List, Iterator

from gooey import GooeyParser

from ..image_preprocess.image_preprocess_keras import (
        image_preprocess_parser,
        image_preprocess,
        ImageDataGenerator,
        )
from .image_classification_generator_config import (
        ImageClassificationGeneratorConfig,
        image_classification_generator_config_parser,
        )
from .directory_dataset.directory_dataset_config import (
        DirectoryDatasetConfig,
        directory_dataset_config_parser,
        #  directory_dataset_config,
        )
from .directory_dataset.directory_dataset import (
        directory_dataset,
        )
from .config_samples import DIR_GEN_CIFAR10
#  from .directory_dataset.config_samles import CIFAR10


def image_classification_generator_parser(
        parser: GooeyParser = GooeyParser(),
        title="Generator",
        image_classification_generator_config=(
            ImageClassificationGeneratorConfig()),
        directory_dataset_config=DirectoryDatasetConfig(),
        dataset_generator_configs=OrderedDict([
            ('directory_cifar10', DIR_GEN_CIFAR10()),
            ]),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    for k, v in dataset_generator_configs.items():
        generator_parser = subs.add_parser(k)
        directory_dataset_config_parser(
                generator_parser, v, True)
        image_classification_generator_config_parser(
                generator_parser, k, v)

    generator_parser = subs.add_parser('directory_generator')
    directory_dataset_config_parser(
            generator_parser,
            directory_dataset_config)
    image_classification_generator_config_parser(
            generator_parser, title,
            image_classification_generator_config)

    return parser


def image_classification_generator(
        cmd: str,
        args: Namespace) -> List[Iterator]:
    #  generator = image_preprocess(args)
    print(cmd, args)
    #  Generator = ImageDataGenerator
    #  print(Generator)

    config = ImageClassificationGeneratorConfig()
    config.update(args)
    config.display()
    generator = ImageDataGenerator(rescale=1./255.)

    if 'directory' in cmd:
        directory_config = directory_dataset(cmd, args)
        directory_config.display()
        generators = [generator.flow_from_directory(
                           directory=directory_config.TRAIN_DIRECTORY,
                           target_size=config.TARGET_SIZE,
                           color_mode=config.COLOR_MODE,
                           class_mode=config.CLASS_MODE,
                           batch_size=config.BATCH_SIZE,
                           shuffle=True,
                           ),
                      (None
                       if directory_config.VAL_DIRECTORY is '' else
                       generator.flow_from_directory(
                           directory=directory_config.TEST_DIRECTORY,
                           target_size=config.TARGET_SIZE,
                           color_mode=config.COLOR_MODE,
                           class_mode=config.CLASS_MODE,
                           batch_size=config.BATCH_SIZE,
                           shuffle=False,
                           ))]
        """
        return directory_generator(generator, args, img_clsf_gen_config)

        if hasattr(args, 'auto_download'):
            if args.auto_download:
                direc_gen_config.auto_download()

        #  return directory_generator(Generator,
        return directory_generator(
                generator,
                direc_gen_config,
        """
    else:
        raise NotImplementedError('wrong dataset_cmd:', cmd)
    return generators
