from pathlib import Path
from typing import Union, Callable
from argparse import ArgumentParser
from gooey import GooeyParser

from keras.applications import Xception as baseModel
from keras.layers import Dense, Flatten
from keras.models import Model

DEFAULT_INPUT_SHAPE = (299, 299, 3)
MODEL_DIR = "checkpoint/Xception/"


def build_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        ) -> Callable:

    feature_layer_parser = parser.add_argument_group(
        title,
        "Input and Feature Extractor",
        gooey_options={'show_border': True, 'columns': 4})
    feature_layer_parser.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        help="Default: {DEFAULT_INPUT_SHAPE}"
    )
    feature_layer_parser.add_argument(
        "--weights",
        choices=['imagenet'],
        default=None,
        metavar="Weights",
        help="Load trained weights."
             "\nDo random initailize if not selected (Ctrl+click)",
    )

    top_layer_parser = parser.add_argument_group(
        "",
        "Top Layers (Classifier)",
        gooey_options={'show_border': True, 'columns': 4})
    top_layer_parser.add_argument(
        "--include_top",
        action='store_true',
        default=False,
        help="lode default top layers")
    top_layer_parser.add_argument(
        "--pooling",
        choices=["flatten", "avg", "max"],
        default="flatten",
        metavar="Flatten or Pooling",
        help="Default: flatten")
    top_layer_parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs='*',
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
    Path(MODEL_DIR).mkdir(exist_ok=True)
    show_and_save_parser.add_argument(
        "--save-path",
        type=str,
        metavar="File path (checkpoint/__/file_name.h5)",
        help="model name to save model",
        gooey_options={
            'validator': {
                'test': "user_input[:len('{MODEL_DIR}')]=='{MODEL_DIR}'",
                'message': 'unvalid save path'
            }
        }
    )
    show_and_save_parser.add_argument(
        "--save-file", type=str,
        metavar="Overwrite File",
        help="model name to save model",
        widget="FileChooser"
    )

    def build(args, baseModel=baseModel):
        if args.input_shape is None:
            args.input_shape = DEFAULT_INPUT_SHAPE
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

        return model
    return build
