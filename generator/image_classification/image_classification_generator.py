from argparse import Namespace
from gooey import GooeyParser

from .image_classification_generator_config import (
        ImageClassificationGeneratorConfig,
        image_classification_generator_config,
        )
from .config_samples import (ICGC_Cifar10,
                             ICGC_Cifar100,
                             ICGC_MNIST,
                             ICGC_FashionMNIST,
                             )
from .directory.directory_generator import (
        DirectoryGeneratorConfig,
        directory_generator_config_parser,
        directory_generator_config,
        directory_generator,
        DGC_Cifar10,
        DGC_Cifar100,
        DGC_MNIST,
        DGC_FashionMNIST,
        )
from ..image_preprocess.image_preprocess_keras import (
        image_preprocess_parser,
        image_preprocess,
        ImageDataGenerator,
        )


def image_classification_generator_parser(
        parser: GooeyParser = GooeyParser(),
        directory_generator_config=(
                DirectoryGeneratorConfig()),
        cifar10_config=DGC_Cifar10(),
        cifar100_config=DGC_Cifar100(),
        mnist_config=DGC_MNIST(),
        fashion_mnist_config=DGC_FashionMNIST(),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    directory_parser = subs.add_parser('directory_check')
    directory_generator_config_parser(
            directory_parser, cifar10_config, auto_download=True)

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


def image_classification_generator(
        cmd: str,
        args: Namespace):
    #  generator = image_preprocess(args)
    print(cmd, args)
    Generator = ImageDataGenerator
    print(Generator)
    #  ICGConfig = image_classification_generator_config(args)
    #  print(ICGConfig().display())

    #  class Config(ImageClassificationGeneratorConfig,
    #               DirectoryGeneratorConfig,):
    #      ...
    if 'directory' in cmd:
        DGConfig = directory_generator_config(args)
        generator_config = DGConfig()
        generator_config.display()
        if hasattr(args, 'auto_download'):
            if args.auto_download:
                generator_config.auto_download()

        return directory_generator(Generator,
                                   generator_config)
    else:
        raise NotImplementedError('wrong dataset_cmd:', cmd)
