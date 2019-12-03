from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from .mrcnn.config import Config
from .fix_validator import fix_validator
from .get_available_gpus import get_available_gpus

RESULT_DIR = Path("results/Mask_RCNN/")


def test_config_parser(
        parser: GooeyParser = GooeyParser(),
        config: Config = Config(),
        modifiable: bool = True,
        ) -> GooeyParser:

    load_parser = parser.add_mutually_exclusive_group(
        'Load Weights')
    load_parser.add_argument(
        '--load_pretrained_weights',
        choices=['coco', 'imagenet', 'last'],
        default='last',
        )
    #  load_parser.add_argument(
    #      '--load_specific_weights',
    #      choices=
    #      )
    #  load_parser.add_argument(
    #      '--load_pretrained_weights',
    #      widget = 'FildChooser'
    #      )

    log_parser = parser.add_argument_group(
        'Log',
        "Save result options",
        gooey_options={'show_border': True, 'columns': 2}
        )
    log_parser.add_argument(
        "--show-image-result",
        metavar="Show Image Results",
        action='store_true',
        default=False,
        )
    log_parser.add_argument(
        "--save-image-result",
        metavar="Save Image Results",
        action='store_true',
        default=False,
        )
    log_parser.add_argument(
        "--result-path", type=str,
        metavar='Result File Path.',
        default=(RESULT_DIR.joinpath('untitled/result.json')
                 if config.NAME is None
                 else RESULT_DIR.joinpath(
                     str(config.NAME)).joinpath('result.json')),
        help='{}{}TIME{}/result.json'.format(
            RESULT_DIR.joinpath('RESULT_NAME'),
            '{', '}')
        )

    return parser


def test_config(args: Namespace) -> Config:
    class TestConfig(Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    return TestConfig()
