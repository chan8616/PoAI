from pathlib import Path
from typing import Union
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from simple_logistic.build import build_parser as simple_
from multilayer_logistic.build import build_parser as multi_


def build_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        description="",
        save_path=""):
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    subs = parser.add_subparsers()

    simple_parser = subs.add_parser('simple_logistic')
    build = simple_(simple_parser)
    parser.set_defaults(simple=build)

    multi_parser = subs.add_parser('multilayer_logistic')
    build = multi_(multi_parser)
    parser.set_defaults(multi=build)

    return parser


if __name__ == "__main__":
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    print(args)
    parser._defaults['build'](args)
