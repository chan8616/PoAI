from collections import OrderedDict
from gooey import Gooey, GooeyParser

from . import model as modellib
from model.keras_applications import build as buildlib
from model.keras_applications.build import build
from .build_config import LogisticConfig
from .config_samples import (LogisticConfig,
                             LogisticMNISTConfig,
                             LogisticOlivettiFacesConfig,
                             )


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        ) -> GooeyParser:
    return buildlib.build_parser(parser,
                                 build_config=LogisticConfig(),
                                 build_configs=OrderedDict([
                                     ('build_mnist', LogisticMNISTConfig()),
                                     ('build_olivetti_faces', LogisticOlivettiFacesConfig()),
                                 ]))
