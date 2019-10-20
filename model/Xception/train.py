from gooey import Gooey, GooeyParser

from ..keras_applications import train as trainlib
#  from ..keras_applications.train import train
from ..keras_applications.config_samples import XceptionConfig


def train_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        ) -> GooeyParser:
    return trainlib.train_parser(config=XceptionConfig())


def train(train_args):
    return trainlib.train(train_args)
