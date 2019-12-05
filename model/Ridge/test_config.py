from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path
from typing import Optional

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore

#  from .train_config import WEIGHTS


class RidgeTestConfig():
    RESULT_NAME = 'untitled'
    TEST_NAME = 'test'

    WEIGHT_PATH = None

    RESULT_DIR = "results/"
    RESLUT_PATH = ""

    def update(self, args: Namespace):
        self.WEIGHT_PATH = args.load_pretrained_weights
        self.RESULT_PATH = args.result_path

    def _parser(self,
                parser: GooeyParser = GooeyParser(),
                ) -> GooeyParser:

        title = 'Test Setting'

        load_parser = parser.add_mutually_exclusive_group()
        #  load_parser.add_argument(
        #      '--load_pretrained_weights',
        #      choices=WEIGHTS,
        #      default=self.WEIGHT,
        #      )
        #  load_parser.add_argument(
        #      '--load_specific_weights',
        #      choices=
        #      )
        load_parser.add_argument(
            '--load_pretrained_weights',
            widget = 'FileChooser',
            )

        log_parser = parser.add_argument_group(
            'Log',
            "Save result options",
            gooey_options={'show_border': True, 'columns': 2}
            )
        log_parser.add_argument(
            "--result-path", type=str,
            metavar='Result File Path.',
            default=Path(self.RESULT_DIR).joinpath(
                         self.RESULT_NAME).joinpath('result.csv'),
            help='{}{}TIME{}/result.json'.format(
                Path(self.RESULT_DIR).joinpath('RESULT_NAME'),
                '{', '}')
            )

        return parser


class RidgeTest():
    def __init__(self, svc_test_config=RidgeTestConfig()):
        self.config = svc_test_config

    def test(self, model, test_generator, stream):
        """Test the model."""
        #  stream.put(('Loding...', None, None)) #  if self.config.WEIGHT in WEIGHTS:
        #      if self.config.WEIGHT == "last":
        #          # find last trained weights
        #          weights_path = model.find_last()
        #      #  else:
        #      #      weights_path = self.config.WEIGHT_PATH

        #      model.load_weights(weights_path, by_name=True)

        if self.config.WEIGHT_PATH:
            model = pickle.load(open(self.config.WEIGHT_PATH, 'rb'))

        stream.put(('Testing', None, None))
        X, Y = test_generator
        result = model.predict(X.values)
        if not Path(self.config.RESULT_PATH).parent.exists():
            Path(self.config.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'prediction': result}).to_csv(
                self.config.RESULT_PATH, index=False)
        stream.put(('end', None, None))
