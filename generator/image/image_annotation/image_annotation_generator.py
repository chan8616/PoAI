from typing import Union
from argparse import ArgumentParser, Namespace
from gooey import Gooey, GooeyParser

from . import coco

CONFIG = {'coco': {}}


def image_annotation_generator_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        config: dict = CONFIG,
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    subs = parser.add_subparsers()

    coco_parser = subs.add_parser('coco')
    coco.coco_parser(coco_parser, CONFIG['coco'])

    return parser


def image_annotation_generator(
        cmd: str,
        args: Namespace):
    if 'coco' == cmd:
        return coco.coco(args)
    else:
        raise NotImplementedError('wrong dataset_cmd:', cmd)


if __name__ == '__main__':
    parser = Gooey(image_annotation_generator_parser)()
    # parser = Parser()
    args = parser.parse_args()
    print(args)
    image_annotation_generator(args)
