# from pathlib import Path
from typing import Union
from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from keras.layers import Dense
from keras.models import Sequential, load_model


def build_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        description="Dimension Setting",
        save_path=""):
    build_parser = parser.add_argument_group(
        title,
        description,
        gooey_options={'columns': 3, 'show_border': True}
    )

    build_parser.add_argument("--input-dim", type=int, default=4)
    build_parser.add_argument("--output-dim", type=int, default=3)
    build_parser.add_argument(
        "--initializer", type=str, default='glorot_uniform')

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

    def build(args):
        model = Sequential()
        model.add(Dense(args.output_dim, input_dim=args.input_dim,
                        activation='softmax',
                        kernel_initializer=args.initializer,
                        ))

        if args.print_model_summary:
            print(model.summary())
        if args.save_path:
            model.save(args.save_path)
        if args.save_file:
            model.save(args.save_file)

        return model

    parser.set_defaults(build=build)

    return build


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





