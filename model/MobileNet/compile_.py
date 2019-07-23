import ast
from typing import Union
from argparse import ArgumentParser, _ArgumentGroup
from argparse import ArgumentTypeError  # , ArgumentError
from gooey import Gooey, GooeyParser

from keras.applications.mobilenet import MobileNet as baseModel
from keras.models import load_model

from model.utils.optimizer import get_optimizer_parser, get_optimizer


def compile_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Compile Options",
        description=""):
    assert isinstance(parser,
                      (ArgumentParser, GooeyParser, _ArgumentGroup)
                      ), type(parser)

    if isinstance(parser, (ArgumentParser, GooeyParser)):
        compile_parser = parser.add_argument_group(
            title=title,
            description=description,
            gooey_options={'show_border': True, 'columns': 2})
    elif isinstance(parser, _ArgumentGroup):
        compile_parser = parser
    else:
        raise ValueError

    compile_parser.add_argument(
        'load_file',
        widget='FileChooser'
    )

    MAX_FREEZE_LAYER = len([layer for layer in baseModel().layers
                            if layer.trainable])

    def max_freeze_layer(value):
        ivalue = ast.literal_eval(value)
        assert isinstance(ivalue, int), ivalue
        if ivalue >= MAX_FREEZE_LAYER or ivalue < 0:
            raise ArgumentTypeError
        return ivalue

    compile_parser.add_argument(
        "--freeze-layer", type=max_freeze_layer, default=0,
        help="freeze layer from bottom (input) / {}".format(MAX_FREEZE_LAYER),
        gooey_options={
            'validation': {
                'test': 'int(user_input) >= {} or int(user_input) < 0'
                        ''.format(MAX_FREEZE_LAYER),
                'message': 'invalid freeze layer number'
            }
        }
    )

    compile_parser.set_defaults(
        max_freeze_layer=max_freeze_layer)

    # detail option
    # compile_parser.add_argument(
    #     "--freeze-layer",
    #     choices=[model.name for model in model.layer if model.trainable],
    #     default=[],
    #     help="/" % len(),
    #     widget='Listbox'
    # )

    compile_parser.add_argument(
        "--loss", choices=["categorical_crossentropy",
                           "sparse_categorical_crossentropy"],
        default="categorical_crossentropy"
    )

    compile_parser.add_argument(
        "--metrics", nargs='*',
        choices=["accuracy"],
        widget='Listbox',
    )

    compile_parser.add_argument(
        "--print-compile-result",
        action='store_true'
    )

    get_optimizer_parser(compile_parser, None, None)

    return parser


def compile_(args):
    model = load_model(args.load_file)

    freeze = args.freeze_layer
    for layer in model.layers:
        if freeze == 0:
            break
        if layer.trainable:
            layer.trainable = False
            freeze -= 1

    model.compile(optimizer=get_optimizer(args),
                  loss=args.loss, metrics=args.metrics)

    if args.print_compile_result:
        print('_________________________________'
              '________________________________')
        print('compiled:', model._is_compiled)
        print('optimizer:', model.optimizer)
        print('loss:', model.loss)
        print('metrics:', model.metrics)
        print('_________________________________'
              '________________________________')
    return model


if __name__ == "__main__":
    # parser = GooeyParser()
    parser = Gooey(compile_parser)(
    )
    args = parser.parse_args()
    print(args)
    parser._defaults['compile_'](args)
