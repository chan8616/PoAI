from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, List, Type

from gooey import Gooey, GooeyParser

#  from keras.layers import Dense, Flatten  # type: ignore
#  from keras.models import Model, load_model  # type: ignore

#  from ..fix_validator import fix_validator
from ..model_config import ModelConfig

LAYERS = OrderedDict([
    ('all', (None, None)),
    ('heads', (-1, None)),
    ])
POOLINGS = 'flatten avg max'.split()


class BuildConfig(ModelConfig):
    NAME = 'Untitled'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = None  # type: ignore

    POOLING = POOLINGS[0]
    #  HIDDEN_LAYERS: List[int]
    HIDDEN_LAYERS = []  # type: ignore
    CLASSES = 0

    def update(self, args: Namespace):
        self.NAME = str(Path(args.log_dir).name)

        self.INPUT_SHAPE = args.input_shape

        self.POOLING = args.pooling
        self.HIDDEN_LAYERS = args.hidden_layers
        self.CLASSES = args.classes


def build_config_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Model",
        build_config=BuildConfig(),
        ) -> GooeyParser:

    feature_layer_parser = parser.add_argument_group(
        title,
        "Input and Feature Extractor",
        gooey_options={'show_border': True, 'columns': 4})
    feature_layer_parser.add_argument(
        "--input_shape", type=eval,
        default=build_config.INPUT_SHAPE,
        help='input_shape',
    )
    #  feature_layer_parser.add_argument(
    #      "--weights", choices=['imagenet'], default=None,
    #      metavar="Weights",
    #      help="Load trained weights."
    #           "\nDo random initailize if not selected (Ctrl+click)",
    #  )

    top_layer_parser = parser.add_argument_group(
        "",
        "Top Layers (Classifier)",
        gooey_options={'show_border': True, 'columns': 4})
    top_layer_parser.add_argument(
        "--pooling",
        choices=POOLINGS,
        default=build_config.POOLING,
        metavar="Flatten or Pooling",
        )
    top_layer_parser.add_argument(
        "--hidden-layers", type=eval,
        metavar="Hidden Layers",
        default=build_config.HIDDEN_LAYERS,
        help="Number of Hidden Layer Nodes",
    )
    top_layer_parser.add_argument(
        "--classes", type=int,
        default=build_config.CLASSES,
        help="Number of Class",
        )

    log_parser = parser.add_argument_group(
        'Log',
        "Show and Save model options",
        gooey_options={'show_border': True, 'columns': 4}
        )
    log_parser.add_argument(
        "--print-model-summary", action='store_true',
        )
    log_parser.add_argument(
        "--log-dir", type=str,
        metavar="Log Directory Path",
        default=Path(build_config.MODEL_DIR).joinpath(
                  str(build_config.NAME)),
        help='{}{}TIME{}/'.format(
            Path(build_config.MODEL_DIR).joinpath('LOG_NAME'),
            '{', '}')
        )

    return parser
