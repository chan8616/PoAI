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

LAYERS = OrderedDict([
    ('all', (None, None)),
    ('heads', (-1, None)),
    ])

KERNELS = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
GAMMAS = ['auto', 'scale']


class SVCBuildConfig(ModelConfig):
    NAME = 'SVC'
    BUILD_NAME = 'build'

    KERNEL = 'rbf'
    DEGREE = 3
    GAMMA: Union[float, str] = 'auto'

    def update(self, args: Namespace):
        self.NAME = str(Path(args.log_dir).name)

        self.KERNAL = args.kernel
        self.DEGREE = args.degree
        try:
            self.GAMMA = float(args.gamma)
        except ValueError:
            if args.gamma in GAMMAS:
                self.GAMMA = args.gamma
            else:
                assert False, 'wrong gamma option {}'.format(args.gamma)
        except Exception as e:
            print(e)
            assert False, 'wrong gamma option {}'.format(args.gamma)

        now = datetime.datetime.now()
        self.LOG_DIR = "{}{:%Y%m%dT%H%M}".format(
                args.log_dir, now)

    def _parser(self,
                parser: GooeyParser = GooeyParser(),
                ) -> GooeyParser:
        title="Build Model"

        kernel_parser = parser.add_argument_group(
            title,
            "Kernel Option",
            gooey_options={'show_border': True, 'columns': 1})
        kernel_parser.add_argument(
            "--kernel", type=str,
            choices=KERNELS,
            metavar='Kernel',
            default=self.KERNEL,
            help='Kernel type to be used in svc.',
        )
        kernel_parser.add_argument(
            "--degree", type=int,
            metavar='Degree',
            default=self.DEGREE,
            help='Degree of the polynomial kernel function.',
        )
        kernel_parser.add_argument(
            "--gamma", type=str,
            metavar='Gamma',
            choices=GAMMAS,
            default=self.GAMMA,
            help='Kernel coefficient for "rbf", "poly" and "sigmoid".'
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
            default=Path(self.MODEL_DIR).joinpath(
                      str(self.NAME)),
            help='{}{}TIME{}/'.format(
                Path(self.MODEL_DIR).joinpath('LOG_NAME'),
                '{', '}')
            )

        return parser
