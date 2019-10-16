from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path

import numpy as np

from .mrcnn.config import Config
from .fix_validator import fix_validator

MODEL_DIR = Path("checkpoint/Mask_RCNN/")


def build_config_parser(
        parser: GooeyParser = GooeyParser(description='Build Option'),
        config: Config = Config(),
        modifiable: bool = True,
        ) -> GooeyParser:

    parser.add_argument(
            '--num_classes',
            type=eval,
            metavar="Classes",
            help="Number of classes (including background).",
            default=config.NUM_CLASSES,
            gooey_options=({} if modifiable else
                           fix_validator(config.NUM_CLASSES))
            )

    backbone_parser = parser.add_argument_group(
        'Backbone')

    backbone_parser.add_argument(
            '--backbone',
            choices=['resnet50', 'resnet101'],
            metavar='Backbone',
            help='Backbone network architecture',
            default=config.BACKBONE,
            gooey_options=({} if modifiable else
                           fix_validator(config.BACKBONE))
            )

    rpn_parser = parser.add_argument_group(
        'RPN',
        'Region Proposal Network',
        )

    rpn_parser.add_argument(
            '--rpn_anchor_scales',
            type=eval,
            metavar='RPN Anchor Scales',
            default=config.RPN_ANCHOR_SCALES,
            help='Length of square anchor side in pixels',
            gooey_options=({} if modifiable else
                           fix_validator(config.RPN_ANCHOR_SCALES))
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
        default=(MODEL_DIR.joinpath('untitled')
                 if config.NAME is None
                 else MODEL_DIR.joinpath(str(config.NAME))),
        help='{}{}TIME{}/'.format(
            MODEL_DIR.joinpath('LOG_NAME'),
            '{', '}')
        )
    #  show_and_save_parser.add_argument(
    #      "--save-path", type=str,
    #      metavar="File path",
    #      default="model.h5",
    #      help="model name to save model",
    #      )
    #  show_and_save_parser.add_argument(
    #      "--save-file", type=str,
    #      metavar="Overwrite File",
    #      help="model name to save model",
    #      widget="FileChooser",
    #      )

    return parser


def build_config(args: Namespace) -> Config:
    class BuildConfig(Config):
        NAME = Path(args.log_dir).name

        NUM_CLASSES = args.num_classes

        BACKBONE = args.backbone

        RPN_ANCHOR_SCALES = args.rpn_anchor_scales

    return BuildConfig()
