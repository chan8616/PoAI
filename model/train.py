from pathlib import Path
from typing import Union, Callable
import sys
sys.path.insert(0, '/home/mlg/yys/project/TensorflowGUI/model')

from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from logistic.simple_logistic.train import train_setting_parser as simple_
from logistic.multilayer_logistic.train import train_setting_parser as multi_
from vgg.vgg16.train import train_setting_parser as vgg16_
from vgg.vgg19.train import train_setting_parser as vgg19_
# from dataset.parser import generator


def train_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser()
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)
    subs = parser.add_subparsers()

    simple_parser = subs.add_parser('simple_logistic')
    train_setting = simple_(simple_parser)
    parser.set_defaults(simple_logistic=train_setting)

    multi_parser = subs.add_parser('multilayer_logistic')
    train_setting = multi_(multi_parser)
    parser.set_defaults(multilayer_logistic=train_setting)

    vgg16_parser = subs.add_parser('vgg16')
    train_setting = vgg16_(vgg16_parser)
    parser.set_defaults(vgg16=train_setting)

    vgg19_parser = subs.add_parser('vgg19')
    train_setting = vgg19_(vgg19_parser)
    parser.set_defaults(vgg19=train_setting)

    return parser


if __name__ == "__main__":
    # parser = Gooey(callbacks_parser)()
    parser = Gooey(train_setting_parser)()
    print(parser._defaults)
    # parser = Gooey(train_parser)()
    args = parser.parse_args()
    print(args)



def train(args1, args2):
    print("in train")
    print(args1, args2)
    model, epochs, callbacks, shuffle = args1
    train_generator, validation_generator = args2

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


def train_parser(
        train_setting_parser: Union[ArgumentParser, GooeyParser,
                                    _ArgumentGroup] = GooeyParser(),
        data_generator_parser: Union[ArgumentParser, GooeyParser,
                                     _ArgumentGroup] = GooeyParser(),
        title="Train Model",
        description=""):

    # modelLoadParser(parser, )
    # compile_parser(parser)
    train_setting_parser(parser)
    data_generator_parser(parser)
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
