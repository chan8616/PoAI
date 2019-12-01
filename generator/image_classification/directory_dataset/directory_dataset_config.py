from pathlib import Path
from typing import Type, Any
from argparse import Namespace

from gooey import GooeyParser
from keras.preprocessing.image import ImageDataGenerator

from ..image_classification_generator_config import ImageClassificationGeneratorConfig


class DirectoryDatasetConfig(ImageClassificationGeneratorConfig):
    NAME = 'Directory'

    def __init__(self):
        super(DirectoryDatasetConfig, self).__init__()
        self.DATASET_DIR = str(Path(self.DATASET_DIR).joinpath(
            Path(self.NAME).name))

        self.TRAIN_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('train'))
        self.TEST_DIRECTORY = str(Path(self.DATASET_DIR).joinpath('test'))

    def update(self, args: Namespace):
        super(DirectoryDatasetConfig, self).update(args)
        self.TRAIN_DIRECTORY = args.directory
        self.TEST_DIRECTORY = args.val_directory

    def save_to_dir(self, x, y, save_dir, save_prefix):
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        if len(list(Path(save_dir).iterdir())) != len(y):
            generator = ImageDataGenerator()
            for _ in generator.flow(
                    x, y,
                    batch_size=len(y),
                    shuffle=False,
                    save_to_dir=save_dir,
                    save_prefix=save_prefix,
                    ):
                print('save_dir: {}\tcount: {}'.format(Path(save_dir), len(y)))
                break

    def auto_download(self):
        print('Downloading...')
        dataset = self.load_data()
        Path(self.TRAIN_DIRECTORY).mkdir(parents=True, exist_ok=True)
        Path(self.TEST_DIRECTORY).mkdir(parents=True, exist_ok=True)
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


def directory_dataset_config_parser(
        parser: GooeyParser,
        directory_dataset_config=(
                DirectoryDatasetConfig()),
        auto_download=False,
        ) -> GooeyParser:

    dir_parser = parser.add_argument_group(
        description='Data folder',
        gooey_options={'columns': 2, 'show_border': True})
    dir_parser.add_argument('--directory', type=str,
                            default=directory_dataset_config.TRAIN_DIRECTORY,
                            metavar='Train Directory',
                            help="data for train",
                            widget='DirChooser')
    if auto_download:
        dir_parser.add_argument(
                '--auto_download',
                metavar='Auto Download',
                action='store_true',
                default=False,
                )
    dir_parser.add_argument('--val-directory', type=str,
                            default=directory_dataset_config.TEST_DIRECTORY,
                            metavar='Validation/Test Directory',
                            help="data for validation with train "
                                 "(optional) or data for test.",
                            widget='DirChooser')

    return parser
