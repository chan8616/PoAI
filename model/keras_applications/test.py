import json
import datetime
import numpy as np
from pathlib import Path

#  from typing import Union, Callable
from argparse import Namespace
from gooey import Gooey, GooeyParser

from .test_config import TestConfig, test_config_parser
from matplotlib import pyplot as plt  # type: ignore

from keras.callbacks import CSVLogger


def test_parser(
        parser: GooeyParser = GooeyParser(),
        title="train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    test_parser = subs.add_parser('test')
    test_config_parser(test_parser)

    return parser


def test(model,
         test_args,
         test_generator,
         #  dataset_test,
         ):
    """Test the model."""
    #  callbacks = [CSVLogger(test_args.result_path)]
    #  results = model.test(test_generator, custom_callbacks=callbacks)
    results = model.test(test_generator,
                         result_save_path=test_args.result_path)
    return

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
    print('saving results to {}...'.format(test_args.result_path))
    with open(test_args.result_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
    print('test complete')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
