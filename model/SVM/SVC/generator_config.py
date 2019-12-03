from typing import Type
from argparse import Namespace
from gooey import Gooey, GooeyParser

import numpy as np

from generator.generator_config import GeneratorConfig


class SVCGeneratorConfig(GeneratorConfig):
    NAME= 'generator'

    DATAFRAME_PATH = ""
    VALID_DATAFRAME_PATH = ""
    X_COL = ["x_col"]
    Y_COL = ["y_col"]

    def update(self, args: Namespace):
        self.DATAFRAME_PATH = args.dataframe_path
        self.VALID_DATAFRAME_PATH = args.valid_dataframe_path
        self.X_COL = args.x_col
        self.Y_COL = args.y_col

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
            help="Dataframe path (file path of .csv format)",
            )
        dataframe_parser.add_argument(
            '--valid_dataframe_path', type=str,
            metavar='DataFrame Path (for valid/test)',
            default=self.VALID_DATAFRAME_PATH,
            help="Dataframe path (file path of .csv format)",
            )
        dataframe_parser.add_argument(
            '--x_col', nargs='*', type=str,
            metavar='Input Data Columns',
            default=np.array2string(np.array(self.X_COL)).strip('[]'),
            help="Input data colum of dataframe.",
            )
        dataframe_parser.add_argument(
            '--y_col', nargs='*', type=str,
            metavar='Target Data Columns',
            default=np.array2string(np.array(self.Y_COL)).strip('[]'),
            help="Target data colum of dataframe.",
            )

        return parser
