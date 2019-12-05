from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf  # type: ignore

from .train_config import WEIGHTS


class TestConfig():
    NAME = None

    WEIGHT = 'last'

    RESULT_DIR = "results/"


def test_config_parser(
        parser: GooeyParser = GooeyParser(),
        title='Test Setting',
        test_config: TestConfig = TestConfig(),
        #  modifiable: bool = True,
        ) -> GooeyParser:

    load_parser = parser.add_mutually_exclusive_group(
        'Load Weights')
    load_parser.add_argument(
        '--load_pretrained_weights',
        choices=WEIGHTS,
        # default=test_config.WEIGHT,
        )
    #  load_parser.add_argument(
    #      '--load_specific_weights',
    #      choices=
    #      )
    load_parser.add_argument(
        '--load_pretrained_file',
        widget='FileChooser'
    )

    log_parser = parser.add_argument_group(
        'Log',
        "Save result options",
        gooey_options={'show_border': True, 'columns': 2}
        )
    log_parser.add_argument(
        "--result-path", type=str,
        metavar='Result File Path.',
        default=(Path(test_config.RESULT_DIR).joinpath('untitled' if test_config.NAME is None
                                                       else str(test_config.NAME))
                 ).joinpath('result.csv'),
        help='{}{}TIME{}/result.csv'.format(
            Path(test_config.RESULT_DIR).joinpath('RESULT_NAME'),
            '{', '}')
        )

    return parser
