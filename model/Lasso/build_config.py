from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, List, Type, Union
import datetime

from gooey import Gooey, GooeyParser

#  from keras.layers import Dense, Flatten  # type: ignore
#  from keras.models import Model, load_model  # type: ignore

#  from ..fix_validator import fix_validator
from model.model_config import ModelConfig

from keras import backend as K


class LassoBuildConfig(ModelConfig):
    NAME = 'Lasso'
    BUILD_NAME = 'build'
    LOG_DIR = str(Path(ModelConfig.MODEL_DIR).joinpath('Lasso'))

    ALPHA = 1.0

    def update(self, args: Namespace):
        self.ALPHA = args.alpha
        self.NAME= str(Path(args.log_dir).name)
        self.LOG_DIR = str(Path(args.log_dir).parent)

    def _parser(self,
                parser: GooeyParser = GooeyParser(),
                ) -> GooeyParser:
        title="Build Model"

        lasso_parser = parser.add_argument_group(
            title,
            'Lasso Regression Option',
            gooey_options={'show_border': True, 'columns': 2}
            )
        lasso_parser.add_argument(
            "--alpha", type=float,
            metavar='Alapha',
            default=self.ALPHA,
            help='Regularization strength.',
            )

        log_parser = parser.add_argument_group(
            'Log',
            "Show and Save model options",
            gooey_options={'show_border': True, 'columns': 2}
            )
        log_parser.add_argument(
            "--print-model", action='store_true',
            )
        log_parser.add_argument(
            "--log-dir", type=str,
            metavar="Log Directory Path",
            default=str(Path(self.MODEL_DIR).joinpath(self.NAME)),
            help='{}{}TIME{}/'.format(
                Path(self.MODEL_DIR).joinpath('LOG_NAME'),
                '{', '}')
            )

        return parser
