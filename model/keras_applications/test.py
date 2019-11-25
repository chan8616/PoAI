import json
import datetime
import numpy as np
from pathlib import Path

#  from typing import Union, Callable
from argparse import Namespace
from gooey import Gooey, GooeyParser

from .test_config import TestConfig, test_config_parser
from matplotlib import pyplot as plt  # type: ignore

from keras.callbacks import CSVLogger


def test_parser(
        parser: GooeyParser = GooeyParser(),
        title="train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    test_parser = subs.add_parser('test')
    test_config_parser(test_parser)

    return parser


def test(model,
         test_args,
         test_generator,
         ):
    """Test the model."""
    results = model.test(test_generator,
                         result_save_path=test_args.result_path)
