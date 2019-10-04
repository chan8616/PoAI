from argparse import Namespace
from gooey import Gooey, GooeyParser

from . import cifar10
from . import directory

CONFIG: dict = {'cifar10': {},
                'directory': {}}


def image_classification_generator_parser(
        parser: GooeyParser = GooeyParser(),
        config: dict = CONFIG,
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    cifar10_parser = subs.add_parser('cifar10')
    cifar10.cifar10_generator_parser(cifar10_parser, CONFIG['cifar10'])

    directory_parser = subs.add_parser('directory')
    directory.directory_generator_parser(
            directory_parser, CONFIG['directory'])

    return parser


def image_classification_generator(
        cmd: str,
        args: Namespace):
    if 'cifar10' == cmd:
        return cifar10.cifar10_generator(args)
    elif 'directory' == cmd:
        return directory.directory_generator(args)
    else:
        raise NotImplementedError('wrong dataset_cmd:', cmd)


if __name__ == '__main__':
    parser = Gooey(image_classification_generator_parser)()
    # parser = Parser()
    args = parser.parse_args()
    print(args)
    image_classification_generator(args)
