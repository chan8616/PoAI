from pathlib import Path
from typing import Type, Any
from argparse import Namespace

from gooey import GooeyParser

from ..dataset import Dataset


class DirectoryDatasetConfig(Dataset):
    NAME = 'Directory'

    def __init__(self):
        super(DirectoryDatasetConfig, self).__init__()
        self.DATASET_DIR = str(Path(self.DATASET_DIR).joinpath(
            Path(self.NAME).name))
        self.DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))

        self.set_log_dir()

    def set_log_dir(self):
        self.TRAIN_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))
        self.VAL_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('valid'))
        self.TEST_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('test'))

    def save_to_dir(self, x, y, save_dir, save_prefix):
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        if len(list(save_dir.iterdir())) != len(y):
            generator = ImageDataGenerator()
            for xy in generator.flow(
                    x, y,
                    batch_size=len(y),
                    shuffle=False,
                    save_dir=save_to_dir,
                    save_prefix=save_prefix,
                    ):
                print('save_dir: {}\tcount: {}'.format(
                    Path(save_dir).name), len(y))
                break

    def auto_download(self):
        print('Downloading...')
        dataset = self.load_data()
        for j, label in enumerate(self.LABELS):
            idx = (dataset[0][1] == j).reshape(-1)
            self.save_to_dir(dataset[0][0][idx],
                             dataset[0][1][idx],
                             str(Path(self.TRAIN_DIRECTORY).joinpath(label)),
                             label,
                             )

            idx = (dataset[1][1] == j).reshape(-1)
            self.save_to_dir(dataset[1][0][idx],
                             dataset[1][1][idx],
                             str(Path(self.TEST_DIRECTORY).joinpath(label)),
                             label,
                             )
        print('Downloading complete!')

    def update(self, args: Namespace):
        self.DIRECTORY = args.directory
        self.VAL_DIRECTORY = args.val_directory
        super(DirectoryDatasetConfig, self).__init__()


def directory_dataset_config_parser(
        parser: GooeyParser,
        directory_dataset_config=(
                DirectoryDatasetConfig()),
        auto_download=False,
        ) -> GooeyParser:

    dir_parser = parser.add_argument_group(
        description='Data folder',
        gooey_options={'columns': 3, 'show_border': True})
    dir_parser.add_argument('directory', type=str,
                            default=directory_dataset_config.DIRECTORY,
                            metavar='Train/Test Directory',
                            help="data for train/test",
                            widget='DirChooser')
    dir_parser.add_argument('--val-directory', type=str,
                            default=directory_dataset_config.VAL_DIRECTORY,
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

    return parser


def directory_dataset_config(
        args: Namespace) -> Type[DirectoryDatasetConfig]:

    class Config(DirectoryDatasetConfig):
        def __init__(self):
            super(DirectoryDatasetConfig, self).__init__()
            self.DIRECTORY = args.directory
            self.VAL_DIRECTORY = args.val_directory

    return Config
