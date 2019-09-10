from pathlib import Path
from typing import Union, Callable
import sys
sys.path.insert(0, '/home/mlg/yys/project/TensorflowGUI/model')

from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from logistic.simple_logistic import test as simple_
from logistic.multilayer_logistic import test as multi_
from vgg.vgg16 import test as vgg16_
from vgg.vgg19 import test as vgg19_
from Xception import test as Xception_
from MobileNet import test as MobileNet_
from InceptionV3 import test as InceptionV3_
from Mask_RCNN import test as Mask_RCNN_
# from dataset.parser import generator


def test_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser()
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)
    subs = parser.add_subparsers()

    simple_parser = subs.add_parser('simple_logistic')
    simple_.test_setting_parser(simple_parser)

    multi_parser = subs.add_parser('multilayer_logistic')
    multi_.test_setting_parser(multi_parser)

    vgg16_parser = subs.add_parser('vgg16')
    vgg16_.test_setting_parser(vgg16_parser)

    vgg19_parser = subs.add_parser('vgg19')
    vgg19_.test_setting_parser(vgg19_parser)

    Xception_parser = subs.add_parser('Xception')
    Xception_.test_setting_parser(Xception_parser)

    MobileNet_parser = subs.add_parser('MobileNet')
    MobileNet_.test_setting_parser(MobileNet_parser)

    InceptionV3_parser = subs.add_parser('InceptionV3')
    InceptionV3_.test_setting_parser(InceptionV3_parser)

    Mask_RCNN_parser = subs.add_parser('Mask_RCNN')
    Mask_RCNN_.test_setting_parser(Mask_RCNN_parser)

    return parser
    # return test_setting
    # return parser


if __name__ == "__main__":
    # parser = Gooey(callbacks_parser)()
    parser = Gooey(test_setting_parser)()
    print(parser._defaults)
    # parser = Gooey(test_parser)()
    args = parser.parse_args()
    print(args)


def test_setting(model_cmd, args):
    if 'simple_logistic' == model_cmd:
        return simple_.test_setting(args)
    elif 'multilayer_logistic' == model_cmd:
        return multi_.test_setting(args)
    elif 'vgg16' == model_cmd:
        return vgg16_.test_setting(args)
    elif 'vgg19' == model_cmd:
        return vgg19_.test_setting(args)
    elif 'Xception' == model_cmd:
        return Xception_.test_setting(args)
    elif 'MobileNet' == model_cmd:
        return MobileNet_.test_setting(args)
    elif 'InceptionV3' == model_cmd:
        return InceptionV3_.test_setting(args)
    elif 'Mask_RCNN' == model_cmd:
        return Mask_RCNN_.test_setting(args)
    else:
        raise NotImplementedError('wrong model_cmd:', model_cmd)


def test(model_cmd, args1, args2):
    if 'simple_logistic' == model_cmd:
        simple_.test(args1, args2)
    elif 'multilayer_logistic' == model_cmd:
        multi_.test(args1, args2)
    elif 'vgg16' == model_cmd:
        vgg16_.test(args1, args2)
    elif 'vgg19' == model_cmd:
        vgg19_.test(args1, args2)
    elif 'Xception' == model_cmd:
        Xception_.test(args1, args2)
    elif 'MobileNet' == model_cmd:
        MobileNet_.test(args1, args2)
    elif 'InceptionV3' == model_cmd:
        InceptionV3_.test(args1, args2)
    elif 'Mask_RCNN' == model_cmd:
        Mask_RCNN_.test(args1, args2)
    else:
        raise NotImplementedError('wrong model_cmd:', model_cmd)


"""
def test_parser(
        test_setting_parser: Union[ArgumentParser, GooeyParser,
                                   _ArgumentGroup] = GooeyParser(),
        data_generator_parser: Union[ArgumentParser, GooeyParser,
                                     _ArgumentGroup] = GooeyParser(),
        title="Train Model",
        description=""):

    # modelLoadParser(parser, )
    # compile_parser(parser)
    test_setting_parser(parser)
    data_generator_parser(parser)
    # parser.set_defaults(test=test)

    def test(model, callbacks,  # test setting output
             epochs, initial_epoch,
             steps_per_epoch, validation_steps,
             test_data, val_data,      # data generator output
             validation_split,
             shuffle,
             ):
        test_data = args.test_data
        val_data = args.val_data

        model.fit_generator(*test_data,
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
"""
