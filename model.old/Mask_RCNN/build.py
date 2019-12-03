from argparse import ArgumentParser
from gooey import Gooey, GooeyParser
from pathlib import Path


try:
    from . import utils
except:
    import utils

import tensorflow as tf

MODEL_DIR = "checkpoint/Mask_RCNN/"


def build_parser(
        parser: GooeyParser = GooeyParser(description='Build Option'),
        ) -> GooeyParser:

    build_parser = parser.add_argument_group(
            'build',
            'Build Option',
            gooey_options={'show_border': True, 'columns': 4})
    build_parser.add_argument(
            '--mode',
            choices=['training', 'inference'],
            default='training',
            metavar="Mode",
            help="Choose model's train/inference mode",
            )
    build_parser.add_argument(
            '--classes', type=eval,
            metavar="Classes",
            help="Number of classes.",
            )
    Path(MODEL_DIR).mkdir(exist_ok=True)
    build_parser.add_argument(
            '--weights', choices=['COCO', 'ImageNet'],
            metavar='Weights',
            default="COCO",
            help="Load trained weights."
                 "\nDo random initailize if not selected (Ctrl+click)",
            )
    build_parser.add_argument(
            '--gpu_count', nargs="*",
            default=[] if tf.test.is_gpu_available() is False else [],
            metavar='gpu_count',
            help="Avaiable gpu list.",
            widget="Listbox",
            )

    """ train parser
    parser.add_argument(
        '--load-weights', type=str,
        metavar='Weight path to load',
        default='{}mask_rcnn_coco.h5'.format(MODEL_DIR),
        gooey_options={
            'validator': {
                'test': "user_input[:len('"+MODEL_DIR+"')]=='"+MODEL_DIR+"'",
                'message': 'unvalid weight path'
                }
            }
        )
    """

    show_and_save_parser = parser.add_argument_group(
        'log',
        "Show and Save model options",
        gooey_options={'show_border': True, 'columns': 4}
        )
    show_and_save_parser.add_argument(
        "--print-model-summary", action='store_true',
        )
    Path(MODEL_DIR).mkdir(exist_ok=True)
    show_and_save_parser.add_argument(
        "--save-path", type=str,
        metavar="File path",
        default="{}model.h5".format(MODEL_DIR),
        help="model name to save model",
        gooey_options={
            'validator': {
                'test': "user_input[:len('"+MODEL_DIR+"')]=='"+MODEL_DIR+"'",
                'message': 'unvalid save path'}},
        )

    show_and_save_parser.add_argument(
        "--save-file", type=str,
        metavar="Overwrite File",
        help="model name to save model",
        widget="FileChooser",
        )

    return parser


def build(args):
    if args.weights == 'COCO':
        if not Path(args.save_path).exists():
            utils.download_trained_weights(args.save_path)
    elif args.weights == 'ImageNet':
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = \
            'https://github.com/fcholletdeep-learning-models/'\
            'releases/download/v0.2/'\
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        get_file(args.save_path,
                 TF_WEIGHTS_PATH_NO_TOP,
                 cache_subdir='models',
                 md5_hash='a268eb855778b3df3c7506639542a6af')

    from .config import Config

    class ModelConfig(Config):
        NAME = args.weights if args.weights is not None else 'Random'
        GPU_COUNT = max(len(args.gpu_count), 1)
    return args.mode, ModelConfig, MODEL_DIR, args


def main():
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    res = build(args)
    print(res)

    """
    if args.weights == 'COCO':
        coco_weight_path = Path(MODEL_DIR).joinpath(args.weights+'.h5')
        if not coco_weight_path.exists():
            utils.download_trained_weights(coco_weight_path)

    if args.mode == 'inference':
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode=args.mode,
                model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    """


if __name__ == '__main__':
    main()

"""
import os
import sys
import random
import math
import numpy as np
#  import skimage.io
#  import matplotlib
#  import matplotlib.pyplot as plt

# Root directory of the project
#  ROOT_DIR = os.path.abspath("../")
CHECKPOINT_DIR = os.path.abspath('checkpoint/Mask_RCNN/')
ROOT_DIR = CHECKPOINT_DIR

# Import Mask RCNN
#  sys.path.append(ROOT_DIR)  # To find local version of the library
from . import model as modellib
from . import visualize
# Import COCO config
#  sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
# To find local version
#  import coco

#  %matplotlib inline

# Directory to save logs and trained model
#  MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#  IMAGE_DIR = os.path.join(ROOT_DIR, "images")


#  class InferenceConfig(coco.CocoConfig):
#      # Set batch size to 1 since we'll be running inference on
#      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#      GPU_COUNT = 1
#      IMAGES_PER_GPU = 1

#  config = InferenceConfig()
#  config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

"""
