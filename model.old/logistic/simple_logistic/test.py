from typing import Union, Callable
from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from keras.models import load_model

# from image_classification import flow_from_dirctory_parser
# from image_classification import image_preprocess
# from image_classification.image_generator import image_generator_parser


def test_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Test Setting",
        description="") -> Callable:

    test_setting_parser = parser.add_argument_group(
        title=title,
        description=description)

    test_setting_parser.add_argument(
        'load_file',
        widget='FileChooser'
    )

    def test_setting(args):
        model = load_model(args.load_file)
        return model

    parser.set_defaults(test_setting=test_setting)

#    compile_parser = parser.add_argument_group(
#        "Compile Parser")
#    compile_parser = compileParser(compile_parser)
#    parser = saveParser(parser)

    return test_setting


def test(args1, args2):
    model = args1
    test_generator = args2
    model.evaluate_generator(test_generator,
                             # callbacks=callbacks,
                             )


if __name__ == "__main__":
    # parser = Gooey(callbacks_parser)()
    model_parser = Gooey(test_setting_parser)()
    model_args = model_parser.parse_args()
    data_parser = Gooey(image_generator_parser)()
    data_args = data_parser.parse_args()
    print(model_args)
    # print(data_ags)

    args1 = model_parser._defaults['test_setting'](model_args)
    args2 = data_parser._defaults['generator'](data_args)
    print(args1)
    print(args2)
    test(args1, args2)