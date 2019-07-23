from typing import Union, Callable
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from model.MobileNet import compile_

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

    get_callbacks_parser(parser)

    return parser
#    compile_parser = parser.add_argument_group(
#        "Compile Parser")
#    compile_parser = compileParser(compile_parser)
#    parser = saveParser(parser)

    # return train_setting


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
                        # validation_split=validation_split,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        shuffle=shuffle,
                        # initial_epoch=initial_epoch,
                        # steps_per_epoch=steps_per_epoch,
                        )


"""
def train_parser(
        train_setting_parser: Union[ArgumentParser, GooeyParser,
                                    _ArgumentGroup] = GooeyParser(),
        data_generator_parser: Union[ArgumentParser, GooeyParser,
                                     _ArgumentGroup] = GooeyParser(),
        # parser: Union[ArgumentParser, GooeyParser,
        #               _ArgumentGroup] = GooeyParser(),
        title="Train Model",
        description=""):

    parser = ArgumentParser()
    # train_setting_parser(parser)
    # image_generator_parser(parser)

    # def train(model, callbacks,  # train setting output
    #           epochs, initial_epoch,
    #           steps_per_epoch, validation_steps,
    #           train_data, val_data,      # data generator output
    #           validation_split,
    #           shuffle,
    #           ):

    def train(train_setting_args, dataset_generator_args):
        model, epochs, callbacks, shuffle = train_setting_parser._defaults[
            'train_setting'](train_setting_args)

        train_generator, validation_generator = \
            dataset_generator_args._defaults[
                'generator'](dataset_generator_args)
        model.fit_generator(train_generator,
                            epochs,
                            callbacks=callbacks,
                            # validation_split=validation_split,
                            validation_data=validation_generator,
                            shuffle=shuffle,
                            # initial_epoch=initial_epoch,
                            # steps_per_epoch=steps_per_epoch,
                            # validation_steps=validation_steps,
                            )
        return
    parser.set_defaults(train=train)
    return parser
"""

if __name__ == "__main__":
    # from image_classification import flow_from_dirctory_parser
    # from image_classification import image_preprocess
    from generator.image_classification.image_generator \
        import image_generator_parser

    # parser = Gooey(callbacks_parser)()
    model_parser = Gooey(train_setting_parser)()
    model_args = model_parser.parse_args()
    data_parser = Gooey(image_generator_parser)()
    data_args = data_parser.parse_args()
    print(model_args)
    # print(data_ags)

    args1 = model_parser._defaults['train_setting'](model_args)
    args2 = data_parser._defaults['generator'](data_args)
    print(args1)
    print(args2)
    train(args1, args2)
