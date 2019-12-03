from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path

from .config import Config

MODEL_DIR = Path("checkpoint/BERT/")


def train_config_parser(
        parser: GooeyParser = GooeyParser(description='Build Option'),
        config: Config = Config(),
        ) -> GooeyParser:
    """
    load_parser = parser.add_mutually_exclusive_group('Load Weights')
    load_parser.add_argument(
        '--load_pretrained_weights',
        choices=['multilingual', 'last'],
        default='multilingual')
    '"""

    steps_parser = parser.add_argument_group(
        title="Train Steps",
        description="Train Steps Setting",
        gooey_options={'columns': 3})

    steps_parser.add_argument(
        "--num_train_epochs", type=int, default=config.NUM_TRAIN_EPOCHS,
        help="number of training per entire dataset"
    )
    steps_parser.add_argument(
        "--train_batch_size", type=int,
        default=config.TRAIN_BATCH_SIZE,
        help="Number of training steps per epoch.",
    )
    steps_parser.add_argument(
        "--predict_batch_size", type=int,
        default=config.PREDICT_BATCH_SIZE,
        help="Number of validation steps to run "
             "at the end of every training epoch.",
    )
    steps_parser.add_argument(
        "--save_checkpoints_steps", type=int,
        default=config.SAVE_CHECKPOINTS_STEPS,
        help="Number of validation steps to run "
             "at the end of every training epoch.",
    )

    compile_parser = parser.add_argument_group(
        title="Compile Settings")
    compile_parser.add_argument(
        '--learning_rate', type=eval,
        metavar='Learning rate',
        default=config.LEARNING_RATE,
    )

    return parser


def train_config(args: Namespace) -> Config:
    class TrainConfig(Config):
        TRAIN_BATCH_SIZE = args.train_batch_size
        NUM_TRAIN_EPOCHS = args.num_train_epochs
        SAVE_CHECKPOINTS_STEPS = args.save_checkpoints_steps

    return TrainConfig()
