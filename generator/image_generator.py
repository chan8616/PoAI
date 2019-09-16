from typing import Union
from argparse import ArgumentParser, Namespace
from gooey import Gooey, GooeyParser

from .image_classification import flow_from_directory as direc_
from .image_annotation import coco


def image_generator_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    subs = parser.add_subparsers()

    coco_parser = subs.add_parser('coco')
    coco.coco_parser(coco_parser)

    dir_parser = subs.add_parser('directory')
    generator = direc_.flow_from_directory_parser(dir_parser)
    parser.set_defaults(directory=generator)

    #  name_parser = subs.add_parser('opendata')
    #  name_parser.add_argument('--name', type=str, choices=['iris'])

    #  file_parser = subs.add_parser('dataframe')
    #  files_parser = file_parser.add_argument_group()
    #  files_parser.add_argument('train-file', type=str,
    #                            metavar='Train File',
    #                            widget='FileChooser')
    #  files_parser.add_argument('validation-file', type=str,
    #                            metavar='Validation File',
    #                            widget='FileChooser')

    return parser


def image_generator(
        dataset_cmd: str,
        args: Namespace):
    if  'coco' == dataset_cmd:
        return coco.coco(args)
    else:
        raise NotImplementedError('wrong dataset_cmd:', model_cmd)



if __name__ == '__main__':
    parser = Gooey(image_generator_parser)()
    # parser = Parser()
    args = parser.parse_args()
    print(args)
    parser._defaults['generator'](args)
