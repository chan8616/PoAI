from argparse import Namespace
from gooey import GooeyParser
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from .mrcnn.config import Config
from .fix_validator import fix_validator
from .get_available_gpus import get_available_gpus


def train_config_parser(
        parser: GooeyParser = GooeyParser(),
        config: Config = Config(),
        modifiable: bool = True,
        ) -> GooeyParser:

    load_parser = parser.add_mutually_exclusive_group(
        'Load Weights')
    load_parser.add_argument(
        '--load_pretrained_weights',
        choices=['coco', 'imagenet', 'last'],
        # default='coco',
        )
    #  load_parser.add_argument(
    #      '--load_specific_weights',
    #      choices=
    #      )
    load_parser.add_argument(
        '--load_pretrained_file',
        widget='FileChooser'
    )

    steps_parser = parser.add_argument_group(
        title="Train Steps",
        description="Train Steps Setting",
        gooey_options={'columns': 3})

    steps_parser.add_argument(
        "epochs", type=int, default=30,
        help="number of training per entire dataset"
    )

    steps_parser.add_argument(
        "--steps_per_epoch", type=int,
        default=config.STEPS_PER_EPOCH,
        help="Number of training steps per epoch.",
    )
    steps_parser.add_argument(
        "--validation_steps", type=int,
        default=config.STEPS_PER_EPOCH,
        help="Number of validation steps to run "
             "at the end of every training epoch.",
    )

    gpu_parser = parser.add_argument_group(
        title='GPU',
        description='GPU Setting',
        gooey_options={'columns': 3})

    gpu_parser.add_argument(
            '--gpu_list', nargs="*",
            choices=get_available_gpus(),
            default=get_available_gpus(),
            metavar='GPU list',
            help="Avaiable GPU list.",
            widget="Listbox",
            )

    gpu_parser.add_argument(
            '--images_per_gpu', type=int,
            default=config.IMAGES_PER_GPU,
            metavar='Images per gpu',
            help="Number of images to train with on each GPU.\n"
                 "A 12GB GPU can typically handle 2 images of 1024x1024px.\n"
                 "Adjust based on your GPU memory and image sizes.\n"
                 "Use the highest number that your GPU can handle "
                 "for best performance.",
            )

    compile_parser = parser.add_argument_group(
            title="Compile Settings")
    compile_parser.add_argument(
            '--learning_rate', type=eval,
            metavar='Learning rate',
            default=config.LEARNING_RATE,
            )

    #  compile_.compile_parser(parser)
    #  get_callbacks_parser(parser)

    return parser


def train_config(args: Namespace) -> Config:
    class TrainConfig(Config):
        STEPS_PER_EPOCH = args.steps_per_epoch
        VALIDATION_STEPS = args.validation_steps
        GPU_COUNT = 1 if not args.gpu_list else len(args.gpu_list)
        IMAGES_PER_GPU = args.images_per_gpu
    return TrainConfig()
