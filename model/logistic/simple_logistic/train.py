from typing import Union, Callable
from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from simple_logistic import compile_
from model.utils.callbacks import get_callbacks_parser, get_callbacks
from simple_logistic.build import MODEL_DIR
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# from image_classification import flow_from_dirctory_parser
# from image_classification import image_preprocess
# from image_classification.image_generator import image_generator

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
        "epochs", type=int, default=10,
        help="number of training per entire dataset"
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

    get_callbacks_parser(parser, model_dir=MODEL_DIR)
    
    return parser


def train_setting(args):
    model = compile_.compile_(args)
    return (model, args.epochs,
            args.epochs if args.validation_steps is None else args.validation_steps,
            get_callbacks(args), args.shuffle)


def train(args1, args2):
    model, epochs, validation_steps, callbacks, shuffle = args1
    train_generator, validation_generator = args2
    model.fit_generator(train_generator,
                        epochs,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        shuffle=shuffle,
                        )
