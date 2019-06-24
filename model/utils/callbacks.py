from typing import Union, Callable
from argparse import ArgumentParser, _ArgumentGroup
from gooey import GooeyParser

from keras.callbacks import ModelCheckpoint, EarlyStopping


def get_callbacks_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title: str = "Callbacks Setting",
        ) -> Callable:

    checkpoint_parser = parser.add_argument_group(
        title=title,
        description="checkpoint callback",
        gooey_options={'columns': 3, 'show_border': True}
    )
    checkpoint_parser.add_argument(
        '--use-checkpoint-callback',
        action='store_true',
    )
    checkpoint_callback_parser(checkpoint_parser)

    earlystopping_parser = parser.add_argument_group(
        title=None,
        description="earlystopping callback",
        gooey_options={'columns': 3, 'show_border': True}
    )
    earlystopping_parser.add_argument(
        '--use-earlystopping-callback',
        action='store_true',
    )
    earlystopping_callback_parser(earlystopping_parser)

    return parser


def get_callbacks(args):
    callbacks = []
    if args.use_checkpoint_callback:
        callbacks.append(
            get_checkpoint_callback(args))

    if args.use_earlystopping_callback:
        callbacks.append(
            get_earlystopping_callback(args))
    return callbacks


def checkpoint_callback_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title: str = "Checkpoint Options",
        description: str = "",
        ) -> Callable:

    if isinstance(parser, (ArgumentParser, GooeyParser)):
        checkpoint_parser = parser.add_argument_group(
            title=title,
            description=description,
            gooey_options={'columns': 3})
    elif isinstance(parser, _ArgumentGroup):
        checkpoint_parser = parser
    else:
        raise ValueError

    checkpoint_parser.add_argument('--ckpt-file-path')
    checkpoint_parser.add_argument(
        '--monitor',
        choices=['acc', 'loss', 'val_loss', 'val_acc'],
        default='loss')
    checkpoint_parser.add_argument('--save-best-only', action='store_true')
    checkpoint_parser.add_argument('--save-weights-only', action='store_true')
    checkpoint_parser.add_argument('--period', type=int, default=1)

    return parser


def get_checkpoint_callback(args):
    return ModelCheckpoint(
        args.ckpt_file_path,
        monitor=args.monitor,
        save_best_only=args.save_best_only,
        save_weights_only=args.save_weights_only,
        period=args.period
        )


def earlystopping_callback_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="EalyStopping Options",
        description="",
        ) -> Callable:

    if isinstance(parser, (ArgumentParser, GooeyParser)):
        earlystopping_parser = parser.add_argument_group(
            title=title,
            description=description,
            gooey_options={'columns': 3})
    elif isinstance(parser, _ArgumentGroup):
        earlystopping_parser = parser

    earlystopping_parser.add_argument('--min_delta')
    earlystopping_parser.add_argument('--patience')
    earlystopping_parser.add_argument('--baseline')
    earlystopping_parser.add_argument('--restore_best_weights')

    return parser


def get_earlystopping_callback(args):
    return EarlyStopping(
        monitor=args.monitor,
        min_delta=args.min_delta,
        patience=args.patience,
        baseline=args.baseline,
        restore_best_weights=args.restore_best_weights
        )
