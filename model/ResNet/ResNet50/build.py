from collections import OrderedDict
from gooey import Gooey, GooeyParser
from pathlib import Path

from . import model as modellib
from model.keras_applications import build as buildlib
from model.keras_applications.build import build
from .config_samples import (ResNet50Config,
                             ResNet50ImagenetConfig,
                             ResNet50CIFAR10Config)


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        ) -> GooeyParser:
    return buildlib.build_parser(parser,
                                 build_config=ResNet50Config(),
                                 build_configs=OrderedDict([
                                     ('build_cifar10', ResNet50CIFAR10Config()),
                                     ('build_imagenet', ResNet50ImagenetConfig()),
                                 ]))


#  def build(mode, build_args):
#      return buildlib.build(model, build_args)
    #  config = buildlib.build_config(build_args)
    #  log_dir = Path(build_args.log_dir).parent

    #  model = modellib.XceptionModel(
    #          config=config,
    #          model_dir=str(log_dir))

    #  if build_args.print_model_summary:
    #      model.keras_model.summary()

    #  return model
