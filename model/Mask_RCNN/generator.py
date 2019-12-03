#  from typing import Union, Callable
from argparse import Namespace
from gooey import Gooey, GooeyParser

from .generator_config import generator_config_parser
from .config_samples import (BalloonConfig, CocoConfig,
                             NucleusConfig, ShapesConfig)
from .dataset_samples import (BalloonDataset, CocoDataset,
                              ViaRegionDataset)
from .utils import Dataset


def generator_parser(
        parser: GooeyParser = GooeyParser(),
        title="Dataset Generator Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    balloon_generator_parser = subs.add_parser('generator_balloon')
    generator_config_parser(balloon_generator_parser,
                            BalloonConfig(),)

    generator_parser = subs.add_parser('generator')
    generator_config_parser(generator_parser, open_dataset=False)

    #  balloon_generator_parser = subs.add_parser('generator_balloon')
    #  generator_config_parser(balloon_generator_parser,
    #                          BalloonConfig(),)

    #  coco_generator_parser = subs.add_parser('generator_coco')
    #  generator_config_parser(coco_generator_parser,
    #                          CocoConfig(),
    #                          years=['2014', '2017'])

    #  nucleus_generator_parser = subs.add_parser('generator_nucleus')
    #  generator_config_parser(nucleus_generator_parser, NucleusConfig())

    #  shapes_generator_parser = subs.add_parser('generator_shapes')
    #  generator_config_parser(shapes_generator_parser, ShapesConfig())

    return parser
    #  model = compile_.compile_(args)
    #  return (model, args.epochs,
    #          args.epochs if args.validation_steps is None
    #          else args.validation_steps,
    #          get_callbacks(args), args.shuffle)


def generator(generator_cmd,
              generator_args):
    """Train the model."""
    if 'balloon' in generator_cmd:
        # Training dataset.
        dataset_train = BalloonDataset()
        dataset_train.load_balloon(generator_args.dataset, "train",
                                   auto_download=generator_args.download)

        # Validation dataset
        dataset_val = BalloonDataset()
        dataset_val.load_balloon(generator_args.dataset, "val",
                                 auto_download=generator_args.download)

        dataset_train.prepare()
        dataset_val.prepare()
    elif 'coco' in generator_cmd:
        # Training dataset.
        dataset_train = CocoDataset()
        dataset_train.load_coco(generator_args.dataset, "train",
                                year=generator_args.year,
                                auto_download=generator_args.download)

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"  # if args.year in '2017' else "minival"
        dataset_val.load_coco(generator_args.dataset, val_type,
                              year=generator_args.year,
                              auto_download=generator_args.download)

        dataset_train.prepare()
        dataset_val.prepare()
    else:
        # Training dataset.
        dataset_train = ViaRegionDataset()
        dataset_train.load_via_region(generator_args.image_directory,
                                      generator_args.annotation_file)

        # Validation dataset
        dataset_val = ViaRegionDataset()
        assert (bool(generator_args.validation_image_directory) ==
                bool(generator_args.validation_annotation_file)), '' \
            'validation setting is not correct.'
        if generator_args.validation_image_directory:
            dataset_val.load_via_region(
                    generator_args.validation_image_directory,
                    generator_args.validation_annotation_file)
        else:
            dataset_val.load_via_region(
                    generator_args.image_directory,
                    generator_args.annotation_file)

        dataset_train.prepare()
        dataset_val.prepare()

    return dataset_train, dataset_val
