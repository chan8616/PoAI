import json
import numpy as np

#  from typing import Union, Callable
from argparse import Namespace
from gooey import Gooey, GooeyParser

from .test_config import test_config_parser, test_config
from .config_samples import (BalloonConfig, CocoConfig,
                             NucleusConfig, ShapesConfig)
from .mrcnn.visualize import display_instances
from matplotlib import pyplot as plt  # type: ignore


def test_parser(
        parser: GooeyParser = GooeyParser(),
        title="train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    #  balloon_test_parser = subs.add_parser('test_balloon')
    #  test_config_parser(balloon_test_parser,
    #                     BalloonConfig(),)

    test_parser = subs.add_parser('test')
    test_config_parser(test_parser)

    #  balloon_train_parser = subs.add_parser('train_balloon')
    #  train_config_parser(balloon_train_parser,
    #                      BalloonConfig(),)

    #  coco_train_parser = subs.add_parser('test_coco')
    #  train_config_parser(coco_train_parser,
    #                      CocoConfig(),)

    #  nucleus_train_parser = subs.add_parser('train_nucleus')
    #  train_config_parser(nucleus_train_parser,
    #                      NucleusConfig(),)

    #  shapes_train_parser = subs.add_parser('train_shapes')
    #  train_config_parser(shapes_train_parser,
    #                      ShapesConfig(),)

    return parser
    #  model = compile_.compile_(args)
    #  return (model, args.epochs,
    #          args.epochs if args.validation_steps is None
    #          else args.validation_steps,
    #          get_callbacks(args), args.shuffle)


def test(model,
         test_args,
         dataset_test):
    # Pick COCO images from the dataset
    image_ids = dataset_test.image_ids

    results = []

    if test_args.save_image_result:
        from pathlib import Path
        Path(test_args.save_image_result).mkdir(parents=True,
                                                exist_ok=True)

    #  image_results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset_test.load_image(image_id)

        print('predict {}/{} images...'.format(
            i+1, len(image_ids)))
        # Run detection
        r = model.detect([image], verbose=0)[0]

        if test_args.show_image_result or test_args.save_image_result:
            fig, ax = plt.subplots()
            display_instances(image, r['rois'],
                              r['masks'], r['class_ids'],
                              dataset_test.class_names,
                              r['scores'],
                              image_id,
                              ax=ax)
            if test_args.show_image_result:
                plt.show(block=False)
                plt.pause(0.01)
            if test_args.save_image_result:
                plt.savefig(test_args.save_image_result +
                            str(image_id) + '.jpg')
            plt.close()

        results += [{k: v.tolist() for k, v in r.items()}]

        # Image result
        #  image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
        #                                     r["rois"], r["class_ids"],
        #                                     r["scores"],
        #                                     r["masks"].astype(np.uint8))
        #  image_results.extend(image_results)
    print('saving results to {}...'.format(test_args.log_file_path))
    with open(test_args.log_file_path, 'w') as f:
        json.dump(results[:1], f, cls=NumpyEncoder)
    print('test complete')

    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
