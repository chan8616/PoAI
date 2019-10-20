#  from typing import Union
from argparse import Namespace
from gooey import Gooey, GooeyParser

from ..image_preprocess import image_preprocess_mask_rcnn


def image_annotation_generator_parser(
        parser: GooeyParser = GooeyParser(),
        ) -> GooeyParser:

    dataset_parser = parser.add_argument_group(
        description='Image annotation dataset',
        gooey_options={'columns': 2, 'show_border': True})
    dataset_parser.add_argument(
            'image-directory', type=str,
            # default="data/cifar10/train",
            metavar='Train/Test Image Directory',
            help="Image directory for train/test",
            widget='DirChooser')
    dataset_parser.add_argument(
            'annotation-file',
            metavar='Train/Test Annotation File',
            help="Image Annotation file for train/test\n"
                 "If blank, just generate images",
            widget='FileChooser')
    dataset_parser.add_argument(
            '--validation-directory', type=str,
            # default="data/cifar10/test",
            metavar='Validation Directory',
            help="data for validation with train "
                 "(optional).",
            widget='DirChooser')
    dataset_parser.add_argument(
            '--validation-annotation-file',
            metavar='Validation Annotation File',
            help="Image Annotation file for validation "
                 "with train (optional).",
            widget='FileChooser')

    image_preprocess_mask_rcnn.image_preprocess_parser(parser)

    return parser


def image_annotation_generator(
        args: Namespace):
    return image_preprocess_mask_rcnn.image_preprocess(args)


if __name__ == '__main__':
    parser = Gooey(image_annotation_generator_parser)()
    # parser = Parser()
    args = parser.parse_args()
    print(args)
    image_annotation_generator(args)
