import ast
from typing import Union
from argparse import ArgumentParser, _ArgumentGroup
from argparse import ArgumentTypeError  # , ArgumentError
from gooey import Gooey, GooeyParser

# from model.vgg.vgg16.optimizer import optimizerParser
# from model.vgg.vgg16.build import build
from keras.applications import VGG19
from keras.models import load_model

from vgg19.optimizer import get_optimizer_parser

MAX_FREEZE_LAYER = len([layer for layer in VGG19().layers if layer.trainable])


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
            gooey_options={'show_border': True, 'columns': 4})
    elif isinstance(parser, _ArgumentGroup):
        compile_parser = parser
    else:
        raise ValueError

    compile_parser.add_argument(
        'load_file',
        widget='FileChooser'
    )

    def max_freeze_layer(value):
        ivalue = ast.literal_eval(value)
        assert isinstance(ivalue, int), ivalue
        if ivalue >= MAX_FREEZE_LAYER or ivalue < 0:
            raise ArgumentTypeError
        return ivalue

    compile_parser.add_argument(
        "--freeze-layer", type=max_freeze_layer, default=0,
        help="freeze layer from bottom (input) / %d" % MAX_FREEZE_LAYER,
    )

    compile_parser.set_defaults(
        MAX_FREEZE_LAYER=MAX_FREEZE_LAYER)


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

    def compile_(args):
        # if args.model_path:
        #     model = args.model_load(args)
        # elif args.base_model:
        #     base_model = args.base_model
        #     model = build(args, base_model)
        # else:
        #     assert False
        # model = VGG16()
        model = load_model(args.load_file)

        freeze = args.freeze_layer
        for layer in model.layers:
            if freeze == 0:
                break
            if layer.trainable:
                layer.trainable = False
                freeze -= 1

        model.compile(optimizer=args.get_optimizer(args),
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

    parser.set_defaults(compile_=compile_)

    return parser


if __name__ == "__main__":
    # parser = GooeyParser()
    parser = Gooey(compile_parser)(
    )
    args = parser.parse_args()
    print(args)
    parser._defaults['compile_'](args)
