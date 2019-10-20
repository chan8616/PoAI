from .image_classification_generator_config import (
        COLOR_MODES,
        ImageClassificationGeneratorConfig,
        )


class ICGC_CIFAR10(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (32, 32)


class ICGC_CIFAR100(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (32, 32)


class ICGC_MNIST(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (28, 28)
    COLOR_MODE = COLOR_MODES[0]


class ICGC_FashionMNIST(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (28, 28)
    COLOR_MODE = COLOR_MODES[0]
