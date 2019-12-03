from argparse import Namespace
from collections import OrderedDict
from typing import Type

from gooey import GooeyParser
#  import tensorflow as tf
#  from tensorflow.python.client import device_lib

from ..fix_validator import fix_validator
from ..get_available_gpus import get_available_gpus
from .build_config import LAYERS

WEIGHTS = ['imagenet', 'last']
LOSSES = ('mean_squared_error binary_crossentropy '
          'categorical_crossentropy sparse_categorical_crossentropy '
          #  'kullback_leibler_divergence '
          ).split()

TRAIN_LAYERS = OrderedDict([(k, v[0]) for k, v in LAYERS.items()])
OPTIMIZERS = ['sgd', 'adam']


class TrainConfig():
    WEIGHT = None

    EPOCHS = 10
    #  VALIDATION_STEPS = 10

    # TRAIN_LAYERS = TRAIN_LAYERS
    TRAIN_LAYER = list(TRAIN_LAYERS.keys())[0]

    LOSS = LOSSES[0]

    OPTIMIZER = OPTIMIZERS[0]
    LEARNING_RATE = .01
    LEARNING_MOMENTUM = .9

    MONITOR = 'loss'

    def __init__(self):
        super(TrainConfig, self).__init__()
        self.TRAIN_LAYERS = TRAIN_LAYERS.copy()

    def update(self, train_args: Namespace):
        self.EPOCHS = train_args.epochs
        #  self.VALIDATION_STEPS = train_args.validation_steps

        self.TRAIN_LAYER = train_args.train_layer

        self.LOSS = train_args.loss

        self.OPTIMIZER = train_args.optimizer
        self.LEARNING_RATE = train_args.learning_rate
        #  self.LEARNING_MOMENTUM = train_args.


def train_config_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        train_config=TrainConfig(),
        #  modifiable: bool = True,
        ) -> GooeyParser:

    load_parser = parser.add_mutually_exclusive_group()
    load_parser.add_argument(
        '--load_pretrained_weights',
        choices=WEIGHTS,
        default=train_config.WEIGHT,
        )
    #  load_parser.add_argument(
    #      '--load_specific_weights',
    #      choices=
    #      )
    #  load_parser.add_argument(
    #      '--load_pretrained_weights',
    #      widget = 'FildChooser'
    #      )

    layers_parser = parser.add_argument_group()
    layers_parser.add_argument(
            '--train-layer',
            choices=list(train_config.TRAIN_LAYERS.keys()),
            default=train_config.TRAIN_LAYER,
            help="Allows selecting wich layers to train."
            )

    steps_parser = parser.add_argument_group(
        title="Train Steps",
        description="Train Steps Setting",
        gooey_options={'columns': 3})

    steps_parser.add_argument(
        "epochs", type=int,
        default=train_config.EPOCHS,
        help="number of training per entire dataset"
    )

    #  steps_parser.add_argument(
    #      "--steps_per_epoch", type=int,
    #      default=config.STEPS_PER_EPOCH,
    #      help="Number of training steps per epoch.",
    #  )
    #  steps_parser.add_argument(
    #      "--validation_steps", type=int,
    #      default=train_config.VALIDATION_STEPS,
    #      help="Number of validation steps to run "
    #           "at the end of every training epoch.",
    #  )

    #  gpu_parser = parser.add_argument_group(
    #      title='GPU',
    #      description='GPU Setting',
    #      gooey_options={'columns': 3})

    #  gpu_parser.add_argument(
    #          '--gpu_list',
    #          # nargs="*",
    #          choices=get_available_gpus(),
    #          #  default=get_available_gpus(),
    #          metavar='GPU list',
    #          help="Avaiable GPU list.",
    #          #  widget="Listbox",
    #          )

    #  gpu_parser.add_argument(
    #          '--images_per_gpu', type=int,
    #          default=train_config.IMAGES_PER_GPU,
    #          metavar='Images per gpu',
    #          help="Number of images to train with on each GPU.\n"
    #               "A 12GB GPU can typically handle 2 images of 1024x1024px."
    #               "Adjust based on your GPU memory and image sizes.\n"
    #               "Use the highest number that your GPU can handle "
    #               "for best performance.",
    #          )

    compile_parser = parser.add_argument_group(
            title="Compile Settings",
            gooey_options={'show_border': True, 'columns': 2})
    compile_parser.add_argument(
        "--loss",
        metavar='Loss',
        choices=LOSSES,
        default=train_config.LOSS,
    )
    compile_parser.add_argument(
            '--optimizer',
            choices=OPTIMIZERS,
            metavar='Optimizer',
            default=train_config.OPTIMIZER,
            )
    compile_parser.add_argument(
            '--learning_rate', type=eval,
            metavar='Learning rate',
            default=train_config.LEARNING_RATE,
            )

    return parser


#  def train_config(train_args: Namespace) -> Type[TrainConfig]:
#      class Config(TrainConfig):
#          EPOCHS = train_args.epochs
#          VALIDATION_STEPS = train_args.validation_steps

#          TRAIN_LAYER = train_args.train_layer

#          LOSS = train_args.loss

#          OPTIMIZER = train_args.optimizer
#          LEARNING_RATE = train_args.learning_rate
#          #  LEARNING_MOMENTUM = train_args.

#      return Config
