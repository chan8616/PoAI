from pathlib import Path
from typing import Union
from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from keras.applications import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model, load_model


def build_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        description="",
        save_path=""):

    feature_layer_parser = parser.add_argument_group(
        title,
        "Input and Feature Extractor",
        gooey_options={'show_border': True, 'columns': 4})
    feature_layer_parser.add_argument(
        "--input_shape", type=int, nargs=3,
        help="Default: 224(>=32) 224(>=32) 3"
    )
    feature_layer_parser.add_argument(
        "--weights", choices=['imagenet'], default=None,
        metavar="Weights",
        help="Load trained weights."
             "\nDo random initailize if not selected (Ctrl+click)",
    )

    top_layer_parser = parser.add_argument_group(
        "",
        "Top Layers (Classifier)",
        gooey_options={'show_border': True, 'columns': 4})
    top_layer_parser.add_argument(
        "--include_top", action='store_true', default=False,
        help="lode default top layers")
    top_layer_parser.add_argument(
        "--pooling", choices=["flatten", "avg", "max"],
        default="flatten", type=str,
        metavar="Flatten or Pooling",
        help="Default: flatten")
    top_layer_parser.add_argument(
        "--hidden-layers", type=int, nargs='*',
        metavar="Number of Hidden Layer Nodes",
        help="Default: no hidden layer"
    )
    top_layer_parser.add_argument(
        "--classes", type=int, default=1000,
        help="Default: 1000")

    show_and_save_parser = parser.add_argument_group(
        "",
        "Show and Save model options",
        gooey_options={'show_border': True, 'columns': 4})
    show_and_save_parser.add_argument(
       "--print-model-summary", action='store_true',
    )
    show_and_save_parser.add_argument(
        "--save-path", type=str,
        metavar="File path (checkpoint/__/file_name.h5)",
        help="model name to save model",
    )
    show_and_save_parser.add_argument(
        "--save-file", type=str,
        metavar="Overwrite File",
        help="model name to save model",
        widget="FileChooser"
    )

    # if save_path is not None:
    #     show_and_save_parser = build_parser.add_argument_group()
    #     show_and_save_parser.add_argument(
    #         "--show-build-result",
    #         action='store_true'
    #     )

    #     def save_model(args):
    #         model = args.model
    #         model.save(parser._defaults["save_path"]+args.save_name)
    #         return model

    # parser = save_model_parser(parser)

    def build(args, baseModel=VGG19):
        if args.input_shape is None:
            args.input_shape = (224, 224, 3)
        if type(args.input_shape) == list:
            args.input_shape = list(224, 224, 3)
        if args.hidden_layers is None:
            args.hidden_layers = []
        base_model = baseModel(include_top=args.include_top,
                               weights=args.weights,
                               input_shape=args.input_shape,
                               pooling=args.pooling,
                               classes=args.classes
                               )
        if not args.include_top:
            x = base_model.output
            if 'flatten' == args.pooling:
                x = Flatten()(x)
            fcs = [Dense(nodes) for nodes in args.hidden_layers]
            for fc in fcs:
                x = fc(x)
            x = Dense(args.classes)(x)
            model = Model(
                inputs=base_model.input,
                outputs=x)
        else:
            model = base_model

        if args.print_model_summary:
            print(model.summary())
        if args.save_path:
            model.save(args.save_path)
        if args.save_file:
            model.save(args.save_file)

        # print(args.save_path)
        # print(args.save_dir)
        # Path(args.save_path + args.save_dir).mkdir(
        #     parents=True, exist_ok=True)
        # model.save(args.save_path + args.save_dir+"/model.h5")
    #    plot_model(model)
        return model

    parser.set_defaults(build=build)

    return parser


if __name__ == "__main__":
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    print(args)
    parser._defaults['build'](args)


def save_model_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Save Model",
        description="",
        save_path="checkpoint/vgg/vgg16/",
        save_name="model.h5"):
    if isinstance(parser, (ArgumentParser, GooeyParser)):
        save_parser = parser.add_argument_group(
            "Save Options", gooey_options={'columns': 2}
        )
    elif isinstance(parser, _ArgumentGroup):
        save_parser = parser
    else:
        raise ValueError
    save_parser.add_argument(
        "--save", action='store_true', default=False)
    # save_parser.add_argument(
    #     '--save_dir', type=str, default=save_path,
    #     metavar="Directory",
    #     help="directory to save model",
    #     # widget='DirChooser',
    # )
    save_parser.set_defaults(save_path=save_path)
    save_parser.add_argument(
        "save-name", type=str,
        default=save_name,
        metavar="Name",
        help="model name to save model",
    )

    def save_model(args):
        model = args.model
        model.save(parser._defaults["save_path"]+args.save_name)
        return model

    return parser


def save_load_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Load Model",
        description="",
        load_path="checkpoint/vgg/vgg16/",
        load_name="model.h5"):
    if isinstance(parser, (ArgumentParser, GooeyParser)):
        load_parser = parser.add_argument_group(
            "Save Options", gooey_options={'columns': 2}
        )
    elif isinstance(parser, _ArgumentGroup):
        load_parser = parser
    else:
        raise ValueError
    # save_parser.add_argument(
    #     '--save_dir', type=str, default=save_path,
    #     metavar="Directory",
    #     help="directory to save model",
    #     # widget='DirChooser',
    # )
    load_parser.set_defaults(load_path=load_path)
    load_parser.add_argument(
        "load-file", type=str,
        default=load_name,
        metavar="Model File",
        help="model name to save model",
    )

    def save_model(args):
        model = load_model(parser._defaults["load_path"]+args.load_name)
        return model

    return parser





