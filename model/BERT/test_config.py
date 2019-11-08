from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path

from .config import Config

MODEL_DIR = Path("checkpoint/BERT/")


def test_config_parser(
        parser: GooeyParser = GooeyParser(description='Build Option'),
        config: Config = Config(),
) -> GooeyParser:

    steps_parser = parser.add_argument_group(
        title="Test Steps",
        description="Test Steps Setting",
        gooey_options={'columns': 3})

    steps_parser.add_argument(
        "--predict_batch_size", type=int,
        default=config.PREDICT_BATCH_SIZE,
        help="Number of validation steps to run "
             "at the end of every training epoch.",
    )

    return parser


def test_config(args: Namespace) -> Config:
    class TrainConfig(Config):
        PREDICT_BATCH_SIZE = args.predict_batch_size

    return TrainConfig()
