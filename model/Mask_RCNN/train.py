from typing import Union, Callable
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

#  from model.Mask_RCNN import 

from model.utils.callbacks import get_callbacks_parser, get_callbacks


def train_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Train Setting",
        description="") -> Callable:

    compile_.compile_parser(parser)
    # compile_parser = parser.add_argument_group(
    #     "Compile Parser")
    # compile_parser = compileParser(compile_parser)

    train_setting_parser = parser.add_argument_group(
        title=title,
        description=description,
        gooey_options={'columns': 3})

    train_setting_parser.add_argument(
        "epoch-schedule", type=eval, nargs='+', default=[40, 120, 160],
        metavar='<train epoch schdule>'
        help="Epoch schedule per each training."
    )
    train_setting_parser.add_argument(
        "--validation_steps", type=int, default=None,
        help="number of steps (batches of samples) to validate before stopping"
    )
    train_setting_parser.add_argument(
        "--shuffle",
        action='store_true',
        default=True
    )

    get_callbacks_parser(parser)

    return parser
