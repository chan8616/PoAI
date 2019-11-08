import os
from typing import List
from argparse import Namespace
from gooey import GooeyParser

import numpy as np

from .config import Config
from .fix_validator import fix_validator


def generator_config_parser(
        parser: GooeyParser = GooeyParser(),
        config: Config = Config(),
        ) -> GooeyParser:

    dataset_parser = parser.add_argument_group(
        title='dataset',
        description='dataset',
        gooey_options={'columns': 2, 'show_border': True})

    dataset_parser.add_argument(
        '--run_file',
        metavar='Train/Test File',
        help="Json file to train/test",
        widget='FileChooser',
        )

    setting_parser = parser.add_argument_group(
        title='data setting',
        description='setting to process data',
        gooey_options={'columns': 3, 'show_border': True}
    )

    setting_parser.add_argument(
        '--do_lower_case',
        action='store_true',
        default=Config.DO_LOWER_CASE
    )

    setting_parser.add_argument(
        "--max_seq_length", type=int,
        default=config.MAX_SEQ_LENGTH,
        help="The maximum total input sequence length after WordPiece tokenization"
    )

    setting_parser.add_argument(
        "--doc_stride", type=int,
        default=config.DOC_STRIDE
    )

    setting_parser.add_argument(
        "--max_query_length", type=int,
        default=config.MAX_QUERY_LENGTH,
    )

    setting_parser.add_argument(
        "--max_answer_length", type=int,
        default=config.MAX_ANSWER_LENGTH,
    )

    return parser


def generator_config(args: Namespace) -> Config:
    class GeneratorConfig(Config):
        TRAIN_FILE = args.run_file
        PREDICT_FILE = args.run_file
        DO_LOWER_CASE = args.do_lower_case
        MAX_SEQ_LENGTH = args.max_seq_length
        DOC_STRIDE = args.doc_stride
        MAX_QUERY_LENGTH = args.max_query_length
        MAX_ANSWER_LENGTH = args.max_answer_length

    return GeneratorConfig()
