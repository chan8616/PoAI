from pathlib import Path
from typing import Type, Any
from argparse import Namespace

from gooey import GooeyParser

from ..image_classification_generator_config import (
        ImageClassificationGeneratorConfig,
        image_classification_generator_config_parser,
        image_classification_generator_config,
        )


class DirectoryGeneratorConfig(ImageClassificationGeneratorConfig):
    NAME = 'Directory'

    def __init__(self):
        super(DirectoryGeneratorConfig, self).__init__()
        self.DATASET_DIR = str(Path(self.DATASET_DIR).joinpath(self.NAME))
        self.DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))
        self.VAL_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('val'))

        self.set_log_dir()

    def set_log_dir(self):
        self.TRAIN_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))
        self.VAL_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('val'))
        self.TEST_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('test'))

    def auto_download(self):
        pass


def directory_generator_config_parser(
        parser: GooeyParser,
        directory_generator_config=(
                DirectoryGeneratorConfig()),
        auto_download=False) -> GooeyParser:

    dir_parser = parser.add_argument_group(
        description='Data folder',
        gooey_options={'columns': 3, 'show_border': True})
    dir_parser.add_argument('directory', type=str,
                            default=directory_generator_config.DIRECTORY,
                            metavar='Train/Test Directory',
                            help="data for train/test",
                            widget='DirChooser')
    dir_parser.add_argument('--val-directory', type=str,
                            default=directory_generator_config.VAL_DIRECTORY,
                            metavar='Validation Directory',
                            help="data for validation with train "
                                 "(optional).",
                            widget='DirChooser')
    if auto_download:
        dir_parser.add_argument(
                '--auto_download',
                metavar='Auto Download',
                action='store_true',
                default=True,
                )

    image_classification_generator_config_parser(
            parser, directory_generator_config)

    return parser


def directory_generator_config(
        args: Namespace) -> Type[DirectoryGeneratorConfig]:
    #  Config_: Type[ImageClassificationGeneratorConfig]
    #  Config_: Any
    Config_ = image_classification_generator_config(args)

    class Config(Config_,
                 DirectoryGeneratorConfig,
                 ):
        def __init__(self):
            super(DirectoryGeneratorConfig, self).__init__()
            self.DIRECTORY = args.directory
            self.VAL_DIRECTORY = args.val_directory

    return Config
