from typing import Union, Callable
from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from multilayer_logistic.compile_ import compile_parser

from keras.callbacks import ModelCheckpoint, EarlyStopping

# from image_classification import flow_from_dirctory_parser
# from image_classification import image_preprocess
# from image_classification.image_generator import image_generator


def callbacks_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Callbacks Setting",):

    checkpoint_parser = parser.add_argument_group(
        title=title,
        description="checkpoint callback",
        gooey_options={'columns': 6, 'show_border': True}
    )
    checkpoint_parser.add_argument(
        '--use-checkpoint-callback',
        action='store_true',
    )
    checkpoint_callback_parser(checkpoint_parser)

    earlystopping_parser = parser.add_argument_group(
        title=None,
        description="earlystopping callback",
        gooey_options={'columns': 5, 'show_border': True}
    )
    earlystopping_parser.add_argument(
        '--use-earlystopping-callback',
        action='store_true',
    )
    earlystopping_callback_parser(earlystopping_parser)

    def get_callbacks(args):
        callbacks = []
        if args.use_checkpoint_callback:
            callbacks.append([
                parser._defaults['get_checkpoint_callback'](args)])

        if args.use_earlystopping_callback:
            callbacks.append([
                parser._defaults['get_earlystopping_callback'](args)])
        return callbacks

    parser.set_defaults(get_callbacks=get_callbacks)
    return parser


def checkpoint_callback_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Checkpoint Options",
        description=""):
    if isinstance(parser, (ArgumentParser, GooeyParser)):
        checkpoint_parser = parser.add_argument_group(
            title=title,
            description=description,
            gooey_options={'columns': 5})
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
    checkpoint_parser.add_argument('--period', type=int)

    def get_checkpoint_callback(args):
        callbacks_ = ModelCheckpoint(
            args.ckpt_file_path,
            monitor=args.monitor,
            save_best_only=args.save_best_only,
            save_weights_only=args.save_weights_only,
            period=args.period
            )
        return callbacks_

    parser.set_defaults(get_checkpoint_callback=get_checkpoint_callback)

    return parser


def earlystopping_callback_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="EalyStopping Options",
        description=""):
    if isinstance(parser, (ArgumentParser, GooeyParser)):
        earlystopping_parser = parser.add_argument_group(
            title=title,
            description=description,
            gooey_options={'columns': 5})
    elif isinstance(parser, _ArgumentGroup):
        earlystopping_parser = parser

    earlystopping_parser.add_argument('--min_delta')
    earlystopping_parser.add_argument('--patience')
    earlystopping_parser.add_argument('--baseline')
    earlystopping_parser.add_argument('--restore_best_weights')

    def get_earlystopping_callback(args):
        callbacks_ = EarlyStopping(
            monitor=args.monitor,
            min_delta=args.min_delta,
            patience=args.patience,
            baseline=args.baseline,
            restore_best_weights=args.restore_best_weights
            )

        return callbacks_

    parser.set_defaults(get_earlystopping_callback=get_earlystopping_callback)

    return parser


def train_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Train Setting",
        description="") -> Callable:

    compile_parser(parser)
    # compile_parser = parser.add_argument_group(
    #     "Compile Parser")
    # compile_parser = compileParser(compile_parser)

    train_setting_parser = parser.add_argument_group(
        title=title,
        description=description)

    train_setting_parser.add_argument(
        "epochs", type=int, default=10,
        help="number of training per entire dataset"
    )

    callbacks_parser(parser)

    def train_setting(args):
        model = parser._defaults['compile_'](args)
        return model, args.epochs, parser._defaults['get_callbacks'](args)

    parser.set_defaults(train_setting=train_setting)

#    compile_parser = parser.add_argument_group(
#        "Compile Parser")
#    compile_parser = compileParser(compile_parser)
#    parser = saveParser(parser)

    return train_setting


"""
def train_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Train Model",
        description=""):

    # modelLoadParser(parser, )
    # compile_parser(parser)
    train_setting_parser(parser)
    image_generator(parser)
    # parser.set_defaults(train=train)

    def train(model, callbacks,  # train setting output
              epochs, initial_epoch,
              steps_per_epoch, validation_steps,
              train_data, val_data,      # data generator output
              validation_split,
              shuffle,
              ):
        train_data = args.train_data
        val_data = args.val_data

        model.fit_generator(*train_data,
                            epochs,
                            callbacks=callbacks,
                            validation_split=validation_split,
                            validation_data=val_data,
                            shuffle=shuffle,
                            initial_epoch=initial_epoch,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            )
        return

    return parser


if __name__ == "__main__":
    # parser = Gooey(callbacks_parser)()
    # parser = Gooey(train_setting_parser)()
    parser = Gooey(train_parser)()
    args = parser.parse_args()
    print(args)

    # parser._defaults['get_callbacks'](args)
"""
