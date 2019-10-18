from argparse import ArgumentParser, ArgumentTypeError, Namespace
import ast

from gooey import Gooey, GooeyParser
from keras.preprocessing.image import ImageDataGenerator


def list_or_number(string):
    val = ast.literal_eval(string)
    acceptable_types = (int, float)
    if any(isinstance(val, x) for x in acceptable_types):
        return val
    if isinstance(val, list):
        numbers = []
        for v in val:
            if any(isinstance(val, v) for x in acceptable_types):
                numbers.append(v)
            else:
                msg = "input float, int or array of those"
                raise ArgumentTypeError(msg)
        return numbers

    msg = "input float, int or array of those"
    raise ArgumentTypeError(msg)


def image_preprocess_parser(
        parser: GooeyParser = GooeyParser(),
        title: str = "Image Preprocessing Options",
        config: dict = {},
        ) -> GooeyParser:

    shift_parser = parser.add_argument_group(
        title,
        "Shift",
        gooey_options={'columns': 3, 'show_border': True})
    shift_parser.add_argument(
        '--featurewise-center',
        action='store_true',
        default=False,
        help="Set input mean 0 over the dataset, feature-wize")
    shift_parser.add_argument(
        '--samplewise_center',
        action='store_true',
        default=False,
        help="Set input mean 0 over the dataset, sample-wize")

    scale_parser = parser.add_argument_group(
        description="Rescale",
        gooey_options={'columns': 3, 'show_border': True})
    scale_parser.add_argument(
        '--featurewise-std-normalization',
        action='store_true',
        default=False,
        help="Divide inputs by std of the dataset, feature-wize")
    scale_parser.add_argument(
        '--samplewise-std-normalization',
        action='store_true',
        default=False,
        help="Divide inputs by std of the dataset, sample-wize")
    scale_parser.add_argument(
        '--rescale', type=lambda x: eval(x), default=0.0,
        help="rescaling factor, apply after all other transformatoins")

    zca_parser = parser.add_argument_group(
        description="zca",
        gooey_options={'columns': 2, 'show_border': True})
    zca_parser.add_argument(
        '--zca-epsilon', type=float, default=1e-6,
        help="epsilon for ZCA whitening")
    zca_parser.add_argument(
        '--zca-whitening',
        action='store_true',
        default=False,
        help="Apply ZCA whitening")

    shift_augment_parser = parser.add_argument_group(
        "Augmentation Options",
        description="Shift augmentation",
        gooey_options={'columns': 2, 'show_border': True})
    shift_augment_parser.add_argument(
        '--width-shift-range', type=list_or_number, default=0.0,
        help="input float or number, or array of those")
    shift_augment_parser.add_argument(
        '--height-shift-range', type=list_or_number, default=0.0,
        help="input float or number, or array of those")

    transform_augment_parser = parser.add_argument_group(
        description="Transform augmentation",
        gooey_options={'columns': 3, 'show_border': True})
    transform_augment_parser.add_argument(
        '--rotation-range', type=int, default=0,
        help="Degree range for random ratations")
    transform_augment_parser.add_argument(
        '--horizontal-flip',
        action='store_true',
        default=False,
        help="Randomly flip inputs horizontally")
    transform_augment_parser.add_argument(
        '--vertical-flip',
        action='store_true',
        default=False,
        help="Randomly flip inputs vertically")

#    parser.add_argument(
#        '--brightness-range', type=float, nargs=2, default=None,
#        help="two floats")

    return parser


def image_preprocess(args: Namespace):
    return ImageDataGenerator(
        featurewise_center=args.featurewise_center,
        samplewise_center=args.samplewise_center,
        featurewise_std_normalization=args.featurewise_std_normalization,
        samplewise_std_normalization=args.samplewise_std_normalization,
        zca_whitening=args.zca_whitening,
        zca_epsilon=args.zca_epsilon,
        rotation_range=args.rotation_range,
        width_shift_range=args.width_shift_range,
        height_shift_range=args.height_shift_range,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        rescale=args.rescale,
        # validation_split=args.validation_split,
    )


# def preprocess_image(args):
#     generator = ImageDataGenerator(
#         featurewise_center=args.featurewise_center,
#         samplewise_center=args.samplewise_center,
#         featurewise_std_normalization=args.featurewise_std_normalization,
#         samplewise_std_normalization=args.samplewise_std_normalization,
#         zca_epsilon=args.zca_epsilon,
#         zca_whitening=args.zca_whitening,
#         rotation_range=args.rotation_range,
#         width_shift_range=args.width_shift_range,
#         height_shift_range=args.height_shift_range,
#         rescale=args.rescale,
#     )
#     return generator


if __name__ == "__main__":
    parser = Gooey(image_preprocess_parser)()
    args = parser.parse_args()
    print(args)
    image_preprocess(args)
