from typing import List
from argparse import Namespace
from gooey import GooeyParser

import numpy as np

from model.Mask_RCNN.mrcnn.config import Config
#  from .fix_validator import fix_validator


def image_preprocess_parser(
        parser: GooeyParser = GooeyParser(),
        title='Image Preprocess',
        config: Config = Config(),
        ) -> GooeyParser:

    image_preprocess_parser = parser.add_argument_group(title=title)

    image_resize_help = \
        "none: No resizing or padding. Return the image unchanged.\n"\
        "square: Resize and pad with zeros to get a square image of size"\
        "[max_dim, max_dim].\n"\
        "pad64: Pads width and height with zeros to make them multiples"\
        "of 64."\
        "If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, "\
        "then it scales up before padding. "\
        "IMAGE_MAX_DIM is ignored in this mode. "\
        "The multiple of 64 is needed to ensure smooth scaling of "\
        "feature maps up and down the 6 levels of the FPN pyramid "\
        "(2**6=64).\n"\
        "crop: Picks random crops from the image. "\
        "First, scales the image based on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, "\
        "then picks a random crop of size IMAGE_MIN_DIM x IMAGE_MIN_DIM. "\
        "Can be used in training only. "\
        "IMAGE_MAX_DIM is not used in this mode."
    image_preprocess_parser.add_argument(
            '--image_resize_mode',
            metavar="Image Resize Mode",
            help=image_resize_help,
            choices=['none', 'square', 'pad64', 'crop'],
            default=config.IMAGE_RESIZE_MODE,
            )

    image_preprocess_parser.add_argument(
            '--image_min_dim', type=int,
            metavar="Image minimum dimension",
            default=config.IMAGE_MIN_DIM,
            )

    image_preprocess_parser.add_argument(
            '--image_max_dim', type=int,
            metavar="Image maximum dimension",
            help="Ignored in pad64 mode",
            default=config.IMAGE_MAX_DIM,
            )

    min_scale_help = \
        "Minimum scaling ratio.\n "\
        "Checked after MIN_IMAGE_DIM and can force further up scaling.\n"\
        "For example, if set to 2 then images are scaled up to double "\
        "the width and height, or more, even if "\
        "MIN_IMAGE_DIM doesn't require it.\n"\
        "However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM."
    image_preprocess_parser.add_argument(
            '--image_min_scale', type=float,
            metavar="Image min scale",
            help=min_scale_help,
            default=str(config.IMAGE_MIN_SCALE),
            )

    image_preprocess_parser.add_argument(
            '--mean_pixel', type=eval,
            metavar="Image Mean (RGB)",
            default=config.MEAN_PIXEL.tolist(),
            )
    return parser


def image_preprocess(args: Namespace) -> Config:
    class GeneratorConfig(Config):
        IMAGE_RESIZE_MODE = args.image_resize_mode

        IMAGE_MIN_DIM = args.image_min_dim
        IMAGE_MAX_DIM = args.image_max_dim
        IMAGE_MIN_SCALE = args.image_min_scale

        MEAN_PIXEL = np.array(args.mean_pixel)

    return GeneratorConfig()
