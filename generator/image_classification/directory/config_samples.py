from pathlib import Path
import shutil

from keras.preprocessing.image import ImageDataGenerator  # type: ignore

from ..config_samples import (ICGC_CIFAR10,
                              ICGC_CIFAR100,
                              ICGC_MNIST,
                              ICGC_FashionMNIST,
                              )
from .directory_generator_config import (
        DirectoryGeneratorConfig,
        )


def flow_save_to_dir(x, y, save_to_dir):
    if not save_to_dir.exists():
        save_to_dir.mkdir(parents=True)
    if len(list(save_to_dir.iterdir())) != len(y):
        generator = ImageDataGenerator()
        for xy in generator.flow(
                x, y,
                batch_size=len(y),
                shuffle=False,
                save_to_dir=str(save_to_dir),
                ):
            print('label: {}\tcount: {}'.format(save_to_dir.name), len(y))
            break


class DGC_CIFAR10(ICGC_CIFAR10,
                  DirectoryGeneratorConfig):
    NAME = 'CIFAR10'
    LABELS = ('airplane automobile bird cat deer '
              'dog frog horse ship truck ').split()

    def __init__(self):
        super(DGC_CIFAR10, self).__init__()
        self.VAL_DIRECTORY = self.TEST_DIRECTORY

    def auto_download(self):
        print('Downloading...')
        from keras.datasets import cifar10  # type: ignore
        dataset = cifar10.load_data()
        for j, label in enumerate(self.LABELS):
            idx = (dataset[0][1] == j).reshape(-1)
            flow_save_to_dir(dataset[0][0][idx],
                             dataset[0][1][idx],
                             Path(self.TRAIN_DIRECTORY).joinpath(label),
                             )

            idx = (dataset[1][1] == j).reshape(-1)
            flow_save_to_dir(dataset[1][0][idx],
                             dataset[1][1][idx],
                             Path(self.TEST_DIRECTORY).joinpath(label),
                             )
        print('Downloading complete!')


class DGC_CIFAR100(ICGC_CIFAR100,
                   DirectoryGeneratorConfig):
    NAME = 'CIFAR100'

    def __init__(self):
        super(DGC_CIFAR100, self).__init__()

    def auto_download(self):
        pass


class DGC_MNIST(ICGC_MNIST,
                DirectoryGeneratorConfig):
    NAME = 'MNIST'

    def __init__(self):
        super(DGC_MNIST, self).__init__()

    def auto_download(self):
        pass


class DGC_FashionMNIST(ICGC_FashionMNIST,
                       DirectoryGeneratorConfig):
    NAME = 'FashionMNIST'

    def __init__(self):
        super(DGC_FashionMNIST, self).__init__()

    def auto_download(self):
        pass
