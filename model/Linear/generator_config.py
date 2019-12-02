from typing import Type
from argparse import Namespace
from gooey import Gooey, GooeyParser

import numpy as np

from generator.generator_config import GeneratorConfig


CLASS_MODES = ['binary', 'categorical', 'multi_output', 'raw', 'sparse']


class LinearGeneratorConfig(GeneratorConfig):
    NAME= 'generator'

    DATAFRAME_PATH = ""
    VALID_DATAFRAME_PATH = ""
    X_COL = ["x_col"]
    Y_COL = ["y_col"]
    CLASS_MODE = CLASS_MODES[3]
    BATCH_SIZE = 32

    def update(self, args: Namespace):
        self.DATAFRAME_PATH = args.dataframe_path
        self.VALID_DATAFRAME_PATH = args.valid_dataframe_path
        self.X_COL = args.x_col
        self.Y_COL = args.y_col
        self.CLASS_MODE = args.class_mode
        self.BATCH_SIZE = args.batch_size

    def _parser(self, 
                parser: GooeyParser = GooeyParser(),
                ) -> GooeyParser:
        title: str = 'DataFrame Generator Options'

        dataframe_parser = parser.add_argument_group(
            description="DataFrame Options",
            gooey_options={'columns': 2, 'show_border': True},
            )
        dataframe_parser.add_argument(
            '--dataframe_path', type=str,
            metavar='DataFrame Path (for train)',
            default=self.DATAFRAME_PATH,
            help="dataframe path (file path of .csv format)",
            )
        dataframe_parser.add_argument(
            '--valid_dataframe_path', type=str,
            metavar='DataFrame Path (for valid/test)',
            default=self.VALID_DATAFRAME_PATH,
            help="dataframe path (file path of .csv format)",
            )
        dataframe_parser.add_argument(
            '--x_col', nargs='*', type=str,
            metavar='Input Data Columns',
            default=np.array2string(np.array(self.X_COL)).strip('[]'),
            help="input data colum of dataframe.",
            )
        dataframe_parser.add_argument(
            '--y_col', nargs='*', type=str,
            metavar='Input Data Columns',
            default=np.array2string(np.array(self.Y_COL)).strip('[]'),
            help="target data colum of dataframe.",
            )
        dataframe_parser.add_argument(
            '--class_mode', type=str,
            choices=CLASS_MODES,
            metavar='Input Data Columns',
            default=self.CLASS_MODE,
            help="target data colum of dataframe.",
            )

        generate_parser = parser.add_argument_group(
            description="Generate Options",
            gooey_options={'columns': 2, 'show_border': True},
            )
        generate_parser.add_argument(
            '--batch-size', type=int,
            metavar='Batch size',
            default=self.BATCH_SIZE,
            )

        return parser
