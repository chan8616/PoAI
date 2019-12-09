from argparse import Namespace
from gooey import Gooey, GooeyParser
from pathlib import Path
from typing import Tuple

from .mrcnn import model as modellib
from .mrcnn.config import Config

#  from .base_build import base_build_parser, build_config
from .build_config import build_config_parser
from .config_samples import (BalloonConfig, CocoConfig,
                             NucleusConfig, ShapesConfig)


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    balloon_build_parser = subs.add_parser('build_balloon')
    build_config_parser(balloon_build_parser, BalloonConfig())

    build_parser = subs.add_parser('build')
    build_config_parser(build_parser)

    #  balloon_build_parser = subs.add_parser('build_balloon')
    #  build_config_parser(balloon_build_parser, BalloonConfig())

    #  coco_build_parser = subs.add_parser('build_coco')
    #  build_config_parser(coco_build_parser, CocoConfig())

    #  nucleus_build_parser = subs.add_parser('build_nucleus')
    #  build_config_parser(nucleus_build_parser, NucleusConfig())

    #  shapes_build_parser = subs.add_parser('build_shapes')
    #  build_config_parser(shapes_build_parser, ShapesConfig())

    return parser


#  def build(build_cmd, build_args):
#      return config(build_args)

#      build_config, args = build_config(config_args)
#      return modellib.MaskRCNN(mode=run_cmd, config=build_config,
#                               model_dir=run_args.model_dir)
#      return build_config(args)


def build(mode,
          build_args,
          build_config,
          train_args,
          run_config,
          generator_config,
          #  model_dir
          ):
    # print('before model config')
    class ModelConfig(Config):
        NAME = build_config.NAME

        NUM_CLASSES = build_config.NUM_CLASSES
        BACKBONE = build_config.BACKBONE
        RPN_ANCHOR_SCALES = build_config.RPN_ANCHOR_SCALES

        STEPS_PER_EPOCH = run_config.STEPS_PER_EPOCH
        VALIDATION_STEPS = run_config.VALIDATION_STEPS
        GPU_COUNT = run_config.GPU_COUNT
        IMAGES_PER_GPU = run_config.IMAGES_PER_GPU

        IMAGE_RESIZE_MODE = generator_config.IMAGE_RESIZE_MODE
        IMAGE_MIN_DIM = generator_config.IMAGE_MIN_DIM
        IMAGE_MAX_DIM = generator_config.IMAGE_MAX_DIM
        IMAGE_MIN_SCALE = generator_config.IMAGE_MIN_SCALE
        MEAN_PIXEL = generator_config.MEAN_PIXEL

        def __init__(self):
            super(ModelConfig, self).__init__()

    log_dir = Path(build_args.log_dir).parent
    if not log_dir.exists():
        log_dir.mkdir()

    config = ModelConfig()
    # print('after model config')
    model = modellib.MaskRCNN(mode=mode,
                              config=config,
                              model_dir=str(log_dir))
    # print('build model')
    if build_args.print_model_summary:
        config.display()
        model.keras_model.summary()
    return model
