from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, List, Type

from gooey import Gooey, GooeyParser

#  from keras.layers import Dense, Flatten  # type: ignore
#  from keras.models import Model, load_model  # type: ignore

#  from ..fix_validator import fix_validator
from model.model_config import ModelConfig

from keras import backend as K

LAYERS = OrderedDict([
    ('all', (None, None)),
    ('heads', (-1, None)),
    ])


class LinearBuildConfig(ModelConfig):
    NAME = 'Linear'
    BUILD_NAME = 'build'

    LOG_DIR = str(Path(ModelConfig.MODEL_DIR).joinpath('Linear'))

    FLATTEN_INPUT_SHAPE = [0]

    HIDDEN_LAYERS = []
    TARGET_SIZE = 1

    def update(self, args: Namespace):
        self.NAME = str(Path(args.log_dir).name)
        self.LOG_DIR = str(Path(args.log_dir).parent)

        self.FLATTEN_INPUT_SHAPE = args.flatten_input_shape

        self.HIDDEN_LAYERS = args.hidden_layers
        self.TARGET_SIZE = args.target_size

    def _parser(self,
                parser: GooeyParser = GooeyParser(),
                ) -> GooeyParser:
        title="Build Model"

        flatten_input_layer_parser = parser.add_argument_group(
            title,
            "Flatten Input Option",
            gooey_options={'show_border': True, 'columns': 2})
        flatten_input_layer_parser.add_argument(
            "--flatten_input_shape", type=eval,
            metavar='Flatten input shape',
            default=self.FLATTEN_INPUT_SHAPE,
            help='Input shape to be flattened.',
        )

        hidden_layer_parser = parser.add_argument_group(
            "",
            "Hidden Layers",
            gooey_options={'show_border': True, 'columns': 2})
        hidden_layer_parser.add_argument(
            "--hidden-layers", type=eval,
            metavar="Hidden layers",
            default=self.HIDDEN_LAYERS,
            help="Number of Hidden Layer Nodes",
        )
        hidden_layer_parser.add_argument(
            "--target-size", type=int,
            metavar='Target size',
            default=self.TARGET_SIZE,
            help="Sizer of regression target.",
            )

        log_parser = parser.add_argument_group(
            'Log',
            "Show and Save model options",
            gooey_options={'show_border': True, 'columns': 2}
            )
        log_parser.add_argument(
            "--print-model-summary", action='store_true',
            )
        log_parser.add_argument(
            "--log-dir", type=str,
            metavar="Log Directory Path",
            default=Path(self.MODEL_DIR).joinpath(
                      str(self.NAME)),
            help='{}{}TIME{}/'.format(
                Path(self.MODEL_DIR).joinpath('LOG_NAME'),
                '{', '}')
            )

        return parser
