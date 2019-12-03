from pathlib import Path
from typing import Union
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from vgg16.build import build_parser as vgg16_
from vgg19.build import build_parser as vgg19_


def build_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        description="",
        save_path=""):
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    subs = parser.add_subparsers()

    vgg16_parser = subs.add_parser('vgg16')
    vgg16_(vgg16_parser)
    parser.set_defaults(vgg16=vgg16_parser._defaults['build'])

    vgg19_parser = subs.add_parser('vgg19')
    vgg19_(vgg19_parser)
    parser.set_defaults(vgg19=vgg19_parser._defaults['build'])

    return parser


if __name__ == "__main__":
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    print(args)
    parser._defaults['build'](args)
