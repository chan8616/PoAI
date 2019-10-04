from argparse import Namespace
from gooey import Gooey, GooeyParser

from .. import image_preprocess

CONFIG: dict = {}


def cifar10_generator_parser(
        parser: GooeyParser = GooeyParser(),
        config: dict = CONFIG,
        ) -> GooeyParser:
    image_preprocess.image_preprocess_parser(parser, config=config)

    generate_parser = parser.add_argument_group(
        description="Generate Options",
        gooey_options={'columns': 2, 'show_border': True})
    generate_parser.add_argument(
        '--batch-size', type=int,
        default=32,
    )
    generate_parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
    )
    return parser


def cifar10_generator(args: Namespace):
    from keras.datasets import cifar10
    dataset = cifar10.load_data()
    generator = [
            image_preprocess.image_preprocess(args).flow(
                dataset[0][0],
                dataset[0][1],
                batch_size=args.batch_size,
                shuffle=args.shuffle),
            image_preprocess.image_preprocess(args).flow(
                dataset[1][0],
                dataset[1][1],
                batch_size=args.batch_size,
                shuffle=args.shuffle),
            ]
    return generator
