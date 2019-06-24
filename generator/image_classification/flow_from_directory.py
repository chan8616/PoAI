from typing import Callable, Union
from argparse import _SubParsersAction, ArgumentParser, _ArgumentGroup
from gooey import GooeyParser

from image_preprocess import image_preprocess_parser


def flow_parser(parser=GooeyParser()):

    pass


def flow_from_directory_parser(
        parser: Union[ArgumentParser, GooeyParser]) -> Callable:

    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    dir_parser = parser.add_argument_group(
        description='Data folder',
        gooey_options={'columns': 2, 'show_border': True})
    dir_parser.add_argument('directory', type=str,
                            # default="data/cifar10/train",
                            metavar='Train/Test Directory',
                            help="data for train/test",
                            widget='DirChooser')
    dir_parser.add_argument('--validation-directory', type=str,
                            # default="data/cifar10/test",
                            metavar='Validation Directory',
                            help="data for validation with train "
                                 "(optional).",
                            widget='DirChooser')

    image_preprocess_parser(parser)
    # dir_parser.add_argument(
    #     'directory',
    #     metavar="Path to target directory",
    #     help="It should contain one subdirectory per class",
    #     widget='FolderChooser'
    # )
    generate_parser = parser.add_argument_group(
        description="Generate Options",
        gooey_options={'columns': 2, 'show_border': True})
    generate_parser.add_argument(
        'target_size', type=int, nargs=2,
        help="Default: 256 256",
    )
    generate_parser.add_argument(
        '--color-mode',
        default='rgb',
        choices='grayscale rgb rgba'.split(),
        help="convert image to 1, 3, or 4 channels",
    )
    # generate_parser.add_argument(
    #     '--classes', nargs='*',
    #     help="",
    # )
    generate_parser.add_argument(
        '--class-mode',
        default='categorical',
        choices=['categorical', 'binary', 'sparse', 'input'],
        help='2D one-hot labels, 1D binary labels, 1D integer labels '
             'or input itself'
    )
    generate_parser.add_argument(
        '--batch-size', type=int,
        default=32,
    )
    generate_parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
    )

    def image_generator(args):
        generator = [dir_parser._defaults[
            'preprocess_image'](args).flow_from_directory(
                args.directory,
                target_size=args.target_size,
                color_mode=args.color_mode,
                class_mode=args.class_mode,
                batch_size=args.batch_size,
                shuffle=args.shuffle)]
        if args.validation_directory:
            generator += [dir_parser._defaults[
                'preprocess_image'](args).flow_from_directory(
                    args.validation_directory,
                    target_size=args.target_size,
                    color_mode=args.color_mode,
                    class_mode=args.class_mode,
                    batch_size=args.batch_size,
                    )]
        else:
            generator += [None]

        return generator
    return image_generator
    # parser.set_defaults(image_generator=image_generator)

    # return parser
