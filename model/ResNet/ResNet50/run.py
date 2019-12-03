from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
#  from ..keras_applications.run import run
from .config_samples import (ResNet50TrainConfig,
                             ResNet50OlivettiFacesConfig,
                             ResNet50ImagenetConfig,
                             ResNet50CIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=ResNet50TrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_olivetti_faces', ResNet50OlivettiFacesConfig()),
                                 ('train_cifar10', ResNet50CIFAR10Config()),
                                 #  ('train_imagenet', ResNet50ImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config, ResNet50TrainConfig())
