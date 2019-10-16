from argparse import Namespace
from gooey import GooeyParser
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from .mrcnn.config import Config
from .fix_validator import fix_validator
from .get_available_gpus import get_available_gpus


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
        default='coco',
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
        gooey_options={'show_border': True, 'columns': 4}
        )
    log_parser.add_argument(
        "--log-file-path", type=str,
        metavar="Log File Path",
        default='predict_result.json',
        help='Log file path',
        )
    log_parser.add_argument(
        "--show-image-result",
        metavar="Show Image Results",
        action='store_true',
        default=False,
        help='Show image result',
        )
    log_parser.add_argument(
        "--save-image-result",
        metavar="Save Image Results",
        default='results/MaskRCNN/',
        help='Directory path to save image result',
        )

    return parser


def test_config(args: Namespace) -> Config:
    class TestConfig(Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    return TestConfig()
