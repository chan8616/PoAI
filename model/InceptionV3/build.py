from collections import OrderedDict
from gooey import Gooey, GooeyParser

from . import model as modellib
from model.keras_applications import build as buildlib
from model.keras_applications.build import build
from .config_samples import (InceptionV3Config,
                             InceptionV3ImagenetConfig,
                             InceptionV3CIFAR10Config,
                             InceptionV3OlivettiFacesConfig)


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        ) -> GooeyParser:
    return buildlib.build_parser(parser,
                                 build_config=InceptionV3Config(),
                                 build_configs=OrderedDict([
                                     ('build_olivetti_faces', InceptionV3OlivettiFacesConfig()),
                                     ('build_cifar10', InceptionV3CIFAR10Config()),
                                     ('build_imagenet', InceptionV3ImagenetConfig()),
                                 ]))


#  def build(mode, build_args):
#      return buildlib.build(model, build_args)
    #  config = buildlib.build_config(build_args)
    #  log_dir = Path(build_args.log_dir).parent

    #  model = modellib.InceptionV3Model(
    #          config=config,
    #          model_dir=str(log_dir))

    #  if build_args.print_model_summary:
    #      model.keras_model.summary()

    #  return model
