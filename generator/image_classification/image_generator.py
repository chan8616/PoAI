from typing import Union
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from flow_from_directory import flow_from_directory_parser


def image_generator_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    # group = group_parser.add_mutually_exclusive_group()
    # group = parser.add_argument_group(gooey_options={'columns': 3})
    # group.add_argument('--name', type=str, choices=[])
    # group.add_argument('--file-path', type=str,
    #                    widget='FileChooser')
    # group.add_argument('--directory-path', type=str,
    #                    widget='DirChooser')

    # train_setting_parser.add_argument(
    #     "--batch-size", type=int, default=32,
    #     help="number of data per model update"
    # )

    subs = parser.add_subparsers()

    dir_parser = subs.add_parser('directory')
    generator = flow_from_directory_parser(dir_parser)
    parser.set_defaults(directory=generator)
    # dir_parser = subs.add_parser('directory')
    # dirs_parser = dir_parser.add_argument_group()
    # dirs_parser.add_argument('train-directory', type=str,
    #                          default="data/cifar10/train",
    #                          metavar='Train Directory',
    #                          widget='DirChooser')
    # dirs_parser.add_argument('validation-directory', type=str,
    #                          default="data/cifar10/test",
    #                          metavar='Validation Directory',
    #                          widget='DirChooser')
    # image_preprocess_parser(dir_parser)
    # dir_parser.add_argument(
    #     'directory',
    #     metavar="Path to target directory",
    #     help="It should contain one subdirectory per class",
    #     widget='FolderChooser'
    # )
    # dir_parser.add_argument(
    #     '--target-size', nargs=2,
    #     help="Default: 256 256",
    # )

    name_parser = subs.add_parser('opendata')
    name_parser.add_argument('--name', type=str, choices=[],)

    file_parser = subs.add_parser('dataframe')
    files_parser = file_parser.add_argument_group()
    files_parser.add_argument('train-file', type=str,
                              metavar='Train File',
                              widget='FileChooser')
    files_parser.add_argument('validation-file', type=str,
                              metavar='Validation File',
                              widget='FileChooser')

    return parser


if __name__ == '__main__':
    parser = Gooey(image_generator_parser)()
    # parser = Parser()
    args = parser.parse_args()
    print(args)
    parser._defaults['generator'](args)
