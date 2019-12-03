import os
from typing import List
from argparse import Namespace
from gooey import GooeyParser

import numpy as np

from .mrcnn.config import Config
from .fix_validator import fix_validator

from generator.image_preprocess import image_preprocess_mask_rcnn


def generator_config_parser(
        parser: GooeyParser = GooeyParser(),
        config: Config = Config(),
        open_dataset: bool = True,
        download: bool = True,
        years: List[str] = [],
        ) -> GooeyParser:

    dataset_parser = parser.add_argument_group(
        title='dataset',
        description='Image annotation dataset',
        gooey_options={'columns': 2, 'show_border': True})

    if open_dataset:
        dataset_parser.add_argument(
                '--dataset',
                metavar='Dataset Folder',
                default=os.path.join("dataset", config.NAME),
                help='Directory of the dataset',
                )
        if download:
            dataset_parser.add_argument(
                    '--download',
                    metavar='Auto Download',
                    help='Automatically download and unzip files',
                    action='store_true',
                    default=True,
                    )
        if years:
            dataset_parser.add_argument(
                    '--year',
                    metavar='year',
                    choices=years,
                    default=years[0])
    else:
        dataset_parser.add_argument(
                'image_directory',
                # default="data/cifar10/train",
                metavar='Train/Test Image Directory',
                help="Image directory for train/test",
                widget='DirChooser',
                )
        dataset_parser.add_argument(
                '--annotation-file',
                metavar='Train/Test Annotation File',
                help="Image Annotation file for train/test\n"
                     "If blank, just generate images",
                widget='FileChooser',
                )
        dataset_parser.add_argument(
                '--validation-image-directory', type=str,
                # default="data/cifar10/test",
                metavar='Validation Directory',
                help="data for validation with train "
                     "(optional).",
                widget='DirChooser')
        dataset_parser.add_argument(
                '--validation-annotation-file',
                metavar='Validation Annotation File',
                help="Image Annotation file for validation "
                     "with train (optional).",
                widget='FileChooser')

    image_preprocess_mask_rcnn.image_preprocess_parser(parser)

    return parser


def generator_config(args: Namespace) -> Config:
    class GeneratorConfig(Config):
        IMAGE_RESIZE_MODE = args.image_resize_mode

        IMAGE_MIN_DIM = args.image_min_dim
        IMAGE_MAX_DIM = args.image_max_dim
        IMAGE_MIN_SCALE = args.image_min_scale

        MEAN_PIXEL = np.array(args.mean_pixel)

    return GeneratorConfig()
