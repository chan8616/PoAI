from collections import OrderedDict
from gooey import Gooey, GooeyParser
from pathlib import Path

#  from . import model as modellib
from ..keras_applications import build as buildlib
from ..keras_applications.build import build
#  from ..keras_applications.config_samples import (BuildConfig,
#                                                   XceptionImagenetConfig)

from .config_samples import (XceptionConfig,
                             XceptionImagenetConfig,
                             XceptionCIFAR10Config)

from keras.applications import Xception  # type: ignore


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        ) -> GooeyParser:
    return buildlib.build_parser(
            parser,
            build_config=XceptionConfig(),
            build_configs=OrderedDict([
                ('build_cifar10', XceptionCIFAR10Config()),
                ('build_imagenet', XceptionImagenetConfig()),
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
