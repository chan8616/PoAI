from pathlib import Path
from typing import Union, Callable
import sys
sys.path.insert(0, '/home/mlg/yys/project/TensorflowGUI/model')

from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from logistic.simple_logistic import train as simple_
from logistic.multilayer_logistic import train as multi_
from vgg.vgg16 import train as vgg16_
from vgg.vgg19 import train as vgg19_
from Xception import train as Xception_
# from dataset.parser import generator


def train_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser()
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)
    subs = parser.add_subparsers()

    simple_parser = subs.add_parser('simple_logistic')
    simple_.train_setting_parser(simple_parser)

    multi_parser = subs.add_parser('multilayer_logistic')
    multi_.train_setting_parser(multi_parser)

    vgg16_parser = subs.add_parser('vgg16')
    vgg16_.train_setting_parser(vgg16_parser)

    vgg19_parser = subs.add_parser('vgg19')
    vgg19_.train_setting_parser(vgg19_parser)

    Xception_parser = subs.add_parser('Xception')
    Xception_train_setting = Xception_.train_setting_parser(Xception_parser)

    def train_setting(cmd, args):
        if 'simple_logistic' == cmd:
            pass
        elif 'Xception' == cmd:
            return Xception_train_setting(args)

    return parser, train_setting
    # return train_setting
    # return parser


if __name__ == "__main__":
    # parser = Gooey(callbacks_parser)()
    parser = Gooey(train_setting_parser)()
    print(parser._defaults)
    # parser = Gooey(train_parser)()
    args = parser.parse_args()
    print(args)


def train_setting(model_cmd, args):
    if 'simple_logistic' == model_cmd:
        train_setting = simple_logistic_.defaults['train_setting'](args)
    elif 'multilayer_logistic' == model_cmd:
        train_setting = simple_logistic_.defaults['train_setting'](args)
    elif 'vgg16' == model_cmd:
        train_setting = simple_logistic_.defaults['train_setting'](args)
    elif 'vgg19' == model_cmd:
        train_setting = simple_logistic_.defaults['train_setting'](args)
    elif 'Xception' == model_cmd:
        train_setting = Xception_.train_setting_parser.defaults['train_setting'](args)
    else:
        raise NotImplementedError('wrong model_cmd:', model_cmd)

    return train_setting(args)


def train(model_cmd, args1, args2):
    if 'simple_logistic' == model_cmd:
        simple_logistic_.train(args1, args2)
    elif 'multilayer_logistic' == model_cmd:
        multilayer_logistic_.train(args1, args2)
    elif 'vgg16' == model_cmd:
        vgg16_.train(args1, args2)
    elif 'vgg19' == model_cmd:
        vgg19_.train(args1, args2)
    elif 'Xception' == model_cmd:
        Xception_.train(args1, args2)
    else:
        raise NotImplementedError('wrong model_cmd:', model_cmd)



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
