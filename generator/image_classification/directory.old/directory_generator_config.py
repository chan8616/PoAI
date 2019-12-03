from pathlib import Path
from typing import Type, Any
from argparse import Namespace

from gooey import GooeyParser

from ...generator_config import (
        GeneratorConfig
)


def flow_save_to_dir(x, y, save_to_dir):
    if not save_to_dir.exists():
        save_to_dir.mkdir(parents=True)
    if len(list(save_to_dir.iterdir())) != len(y):
        generator = ImageDataGenerator()
        for xy in generator.flow(
                x, y,
                batch_size=len(y),
                shuffle=False,
                save_to_dir=str(save_to_dir),
                ):
            print('label: {}\tcount: {}'.format(save_to_dir.name), len(y))
            break


class DirectoryGeneratorConfig(GeneratorConfig):
    NAME = 'Directory'

    def __init__(self):
        super(DirectoryGeneratorConfig, self).__init__()
        self.DATASET_DIR = str(Path(self.DATASET_DIR).joinpath(self.NAME))
        self.DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))

        self.set_log_dir()

    def set_log_dir(self):
        self.TRAIN_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))
        self.VAL_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('valid'))
        self.TEST_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('test'))

    def auto_download(self, dataset, labels):
        #  print('Downloading...')
        #  dataset = cifar10.load_data()
        assert len(np.unique(dataset[0][1])) == len(labels)
        for j, label in enumerate(self.LABELS):
            idx = (dataset[0][1] == j).reshape(-1)
            flow_save_to_dir(dataset[0][0][idx],
                             dataset[0][1][idx],
                             Path(self.TRAIN_DIRECTORY).joinpath(label),
                             )

            idx = (dataset[1][1] == j).reshape(-1)
            flow_save_to_dir(dataset[1][0][idx],
                             dataset[1][1][idx],
                             Path(self.TEST_DIRECTORY).joinpath(label),
                             )
        print('Downloading complete!')


def directory_generator_config_parser(
        parser: GooeyParser,
        directory_generator_config=(
                DirectoryGeneratorConfig()),
        ) -> GooeyParser:

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
    #  if auto_download:
    #      dir_parser.add_argument(
    #              '--auto_download',
    #              metavar='Auto Download',
    #              action='store_true',
    #              default=True,
    #              )

    #  image_classification_generator_config_parser(
    #          parser, directory_generator_config)

    return parser


def directory_generator_config(
        args: Namespace) -> Type[DirectoryGeneratorConfig]:

    class Config(DirectoryGeneratorConfig):
        def __init__(self):
            super(DirectoryGeneratorConfig, self).__init__()
            self.DIRECTORY = args.directory
            self.VAL_DIRECTORY = args.val_directory

    return Config
