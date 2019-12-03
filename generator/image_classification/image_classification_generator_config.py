from typing import Type
from argparse import Namespace
from gooey import Gooey, GooeyParser

from ..generator_config import GeneratorConfig


COLOR_MODES = 'grayscale rgb rgba'.split()
CLASS_MODES = 'categorical binary sparse input'.split()


class ImageClassificationGeneratorConfig(GeneratorConfig):
    TARGET_SIZE = (256, 256)
    COLOR_MODE = COLOR_MODES[1]
    CLASS_MODE = CLASS_MODES[0]
    BATCH_SIZE = 32
    SHUFFLE = True

    def update(self, args: Namespace):
        self.TARGET_SIZE = args.target_size
        self.COLOR_MODE = args.color_mode
        self.CLASS_MODE = args.class_mode
        self.BATCH_SIZE = args.batch_size


def image_classification_generator_config_parser(
        parser: GooeyParser,
        title: str = 'Image Classification Generator Options',
        image_classification_generator_config=(
            ImageClassificationGeneratorConfig()),
        ) -> GooeyParser:

    generate_parser = parser.add_argument_group(
        description="Generate Options",
        gooey_options={'columns': 2, 'show_border': True})
    generate_parser.add_argument(
        '--target_size', type=eval,
        metavar='Target size',
        default=image_classification_generator_config.TARGET_SIZE,
    )
    generate_parser.add_argument(
        '--color-mode',
        metavar='Color mode',
        choices=COLOR_MODES,
        default=image_classification_generator_config.COLOR_MODE,
        help="convert image to 1, 3, or 4 channels",
    )
    generate_parser.add_argument(
        '--class-mode',
        metavar='Class mode',
        choices=CLASS_MODES,
        default=image_classification_generator_config.CLASS_MODE,
        help='2D one-hot labels, 1D binary labels, 1D integer labels '
             'or input itself'
    )
    generate_parser.add_argument(
        '--batch-size', type=int,
        metavar='Batch size',
        default=image_classification_generator_config.BATCH_SIZE,
    )
    #  generate_parser.add_argument(
    #      '--shuffle',
    #      action='store_true',
    #      default=True,
    #      help='Whether to shuffle the data (default: True).'
    #           'If set to False, sorts the data in alphanumeric order.'
    #  )

    return parser


"""
def image_classification_generator_config(
        args: Namespace) -> Type[ImageClassificationGeneratorConfig]:

    class Config(ImageClassificationGeneratorConfig):
        TARGET_SIZE = args.target_size
        COLOR_MODE = args.color_mode
        CLASS_MODE = args.class_mode
        BATCH_SIZE = args.batch_size

    return Config
"""
