import json
import datetime
import numpy as np
from pathlib import Path

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
         dataset_test,
         stream=None):
    # Pick COCO images from the dataset
    image_ids = dataset_test.image_ids

    results = []

    #  now = datetime.datetime.now()
    #  result_dir = Path("{}{:%Y%m%dT%H%M}".format(
    #          str(Path(test_args.result_path).parent), now))
    #  if not result_dir.exists():
    #      result_dir.mkdir(parents=True)
    result_path = Path(model.result_dir).joinpath(
            Path(test_args.result_path).name)

    #  image_results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset_test.load_image(image_id)

        print('predict {}/{} images...'.format(
            i+1, len(image_ids)))
        # Run detection
        r = model.detect([image], verbose=0)[0]

        if test_args.show_image_result or test_args.save_image_result:
            image_name = dataset_test.image_info[image_id]['id']
            fig, ax = plt.subplots()
            display_instances(image, r['rois'],
                              r['masks'], r['class_ids'],
                              dataset_test.class_names,
                              r['scores'],
                              image_name,
                              ax=ax)
            if test_args.show_image_result:
                plt.show(block=False)
                plt.pause(1)
            if test_args.save_image_result:
                image_save_path = model.result_dir.joinpath(image_name)
                print('saving image to {}...'.format(image_save_path))
                plt.savefig(str(image_save_path))
            plt.close()

        results += [{k: v.tolist() for k, v in r.items()}]

        # Image result
        #  image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
        #                                     r["rois"], r["class_ids"],
        #                                     r["scores"],
        #                                     r["masks"].astype(np.uint8))
        #  image_results.extend(image_results)
    print('saving results to {}...'.format(result_path))
    with open(str(result_path), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
    print('test complete')

    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
