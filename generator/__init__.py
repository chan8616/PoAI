import os
from pathlib import Path
from treelib import Tree
import sys
sys.path.insert(
    0, os.path.join(os.getcwd(), 'generator'))

import image_classification
IMAGE_CLASSIFICATION_TREE = Tree()
IMAGE_CLASSIFICATION_TREE.create_node('image classification',
                                      data=image_classification)
import iris
IRIS_TREE = Tree()
IRIS_TREE.create_node('iris dataset',
                       data=iris)

"""
class DATASET():
    def __init__(self, name, path):
        self.spec = {'name': name,
                'path': path,
                }


class PROVIDER(DATASET):
    def __init__(self, *args, **kwargs):
        super(PROVIDER, self).__init__(args, kwargs)

class IMAGEPROVIDER(PROVIDER):
    def __init__(self, *args, **kwargs):
        super(IMAGEPROVIDER, self).__init__(args, kwargs)

from dataset.image_classification import IMAGECLASSIFICATIONDICT
from dataset import image_classification
from dataset.image_classification import MNIST_TREE
from dataset.image_classification import FLOW_FROM_DIRECTORY_TREE

DATASETDICT = {
    'image_classification': IMAGECLASSIFICATIONDICT,
    'mnist': None,
}

IMAGE_CLASSIFICATION_TREE = Tree()
IMAGE_CLASSIFICATION_TREE.create_node('image classification',
                                      data=image_classification)
IMAGE_CLASSIFICATION_TREE.paste(IMAGE_CLASSIFICATION_TREE.root, MNIST_TREE)
IMAGE_CLASSIFICATION_TREE.paste(IMAGE_CLASSIFICATION_TREE.root,
                                FLOW_FROM_DIRECTORY_TREE)
"""
