#  from typing import Union, Callable
from argparse import Namespace
from gooey import GooeyParser

from .bert.run_squad import read_squad_examples
from .generator_config import generator_config_parser


def generator_parser(
        parser: GooeyParser = GooeyParser(),
        title="Dataset Generator Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    generator_parser_ = subs.add_parser('generator')
    generator_config_parser(generator_parser_)

    return parser


def generator(generator_cmd,
              generator_args):
    """Train the model."""
    if 'KorQuAD' in generator_cmd:
        # Training dataset.
        dataset_train = read_squad_examples(input_file=generator_args.run_file, is_training=True)

        # Validation dataset
        dataset_val = read_squad_examples(input_file=generator_args.run_file, is_training=False)

    else:
        print("ERROR!!!")
        # Exception
        # Training dataset.
        dataset_train = read_squad_examples(input_file=generator_args.run_file, is_training=True)

        # Validation dataset
        dataset_val = read_squad_examples(input_file=generator_args.run_file, is_training=False)

    return dataset_train, dataset_val
