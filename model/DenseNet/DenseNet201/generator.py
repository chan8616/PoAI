from collections import OrderedDict
from gooey import GooeyParser

from generator.image_classification.image_classification_generator \
        import image_classification_generator_parser
from generator.image_classification.image_classification_generator \
        import image_classification_generator as generator
from model.keras_applications.generator_config_samples \
        import ImageClassificationGeneratorConfig, DirectoryDatasetConfig

from .config_samples import (
        DenseNet201CIFAR10Config,
        DenseNet201OlivettiFacesConfig,
        )


def generator_parser(
        parser: GooeyParser = GooeyParser(),
        title="Generator Setting",
        ) -> GooeyParser:
    return image_classification_generator_parser(
            parser,
            title,
            image_classification_generator_config=(
                ImageClassificationGeneratorConfig()),
            directory_dataset_config=DirectoryDatasetConfig(),
            dataset_generator_configs=OrderedDict([
                ('directory_olivetti_faces', DenseNet201OlivettiFacesConfig()),
                ('directory_cifar10', DenseNet201CIFAR10Config()),
                #  'directory_cifar100': DGC_CIFAR100(),
                #  'directory_mnist': DGC_MNIST(),
                #  'directory_fashion_mnist': DGC_FashionMNIST(),
                ]))
