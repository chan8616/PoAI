from pathlib import Path
from typing import Type, Any
from argparse import Namespace

from gooey import GooeyParser

from ..image_classification_generator_config import (
        ImageClassificationGeneratorConfig,
        image_classification_generator_config_parser,
        image_classification_generator_config,
        )


class DataframeGeneratorConfig(ImageClassificationGeneratorConfig):
    NAME = 'Dataframe'

    def __init__(self):
        super(DataframeGeneratorConfig, self).__init__()
        self.DATASET_DIR = str(
                Path(self.DATASET_DIR).joinpath(self.NAME))
        self.DIRECTORY = str(
                Path(self.DATASET_DIR).joinpath('images'))
        self.DATAFRAME = str(
                Path(self.DATASET_DIR).joinpath('train.csv'))

        self.set_log_dir()

    def set_log_dir(self):
        self.TRAIN_DATAFRAME = str(
                Path(self.DATASET_DIR).joinpath('train.csv'))
        self.VAL_DATAFRAME = str(
                Path(self.DATASET_DIR).joinpath('valid.csv'))
        self.TEST_DATAFRAME = str(
                Path(self.DATASET_DIR).joinpath('test.csv'))

    def auto_download(self):
        pass


def dataframe_generator_config_parser(
        parser: GooeyParser,
        dataframe_generator_config=(
                DataframeGeneratorConfig()),
        auto_download=False) -> GooeyParser:

    dir_parser = parser.add_argument_group(
        description='Dataframe file',
        gooey_options={'columns': 2, 'show_border': True})
    dir_parser.add_argument('dataframe', type=str,
                            default=dataframe_generator_config.DATAFRAME,
                            metavar='Train/Test Dataframe',
                            help=""
                            "Pandas dataframe containing the filepaths "
                            "relative to directory of the images "
                            "in a string column.",
                            widget='FileChooser')
    dir_parser.add_argument('directory', type=str,
                            default=dataframe_generator_config.DIRECTORY,
                            metavar='Train/Test Image Directory',
                            help="path to the directory to read images from.",
                            widget='FileChooser')
    dir_parser.add_argument('--val-dataframe', type=str,
                            default=dataframe_generator_config.VAL_DATAFRAME,
                            metavar='Validation Dataframe',
                            help="(optional).",
                            widget='FileChooser')
    dir_parser.add_argument('--val-directory', type=str,
                            default=dataframe_generator_config.DIRECTORY,
                            metavar='Vailidation Image Directory',
                            help="(optional).",
                            widget='FileChooser')
    if auto_download:
        dir_parser.add_argument(
                '--auto_download',
                metavar='Auto Download',
                action='store_true',
                default=True,
                )

    image_classification_generator_config_parser(
            parser, dataframe_generator_config)

    return parser


def dataframe_generator_config(
        args: Namespace) -> Type[DataframeGeneratorConfig]:
    #  Config_: Type[ImageClassificationGeneratorConfig]
    #  Config_: Any
    Config_ = image_classification_generator_config(args)

    class Config(Config_,
                 DataframeGeneratorConfig,
                 ):
        def __init__(self):
            super(DataframeGeneratorConfig, self).__init__()
            self.DATAFRAME = args.dataframe
            self.DIRECTORY = args.directory
            self.VAL_DATAFRAME = args.val_dataframe
            self.VAL_DIRECTORY = args.val_directory

    return Config
