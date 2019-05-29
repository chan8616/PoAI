import os
from pathlib import Path
from treelib import Tree
import sys
sys.path.insert(0, '/home/mlg/yys/project/TensorflowGUI/model')

import logistic
from logistic import SIMPLE_LOGISTIC_TREE
from logistic import MULTILAYER_LOGISTIC_TREE

import vgg
from vgg import VGG16_TREE
from vgg import VGG19_TREE


LOGISTIC_TREE = Tree()
LOGISTIC_TREE.create_node(
    'logistic',
    Path('TensorflowGUI/model/logistic/'), data=logistic)
LOGISTIC_TREE.paste(LOGISTIC_TREE.root, SIMPLE_LOGISTIC_TREE)
LOGISTIC_TREE.paste(LOGISTIC_TREE.root, MULTILAYER_LOGISTIC_TREE)

VGG_TREE = Tree()
VGG_TREE.create_node(
    'vgg',
    # 'model/vgg/',
    Path(vgg.__path__[0]).relative_to(os.getcwd()),
    data=vgg)
VGG_TREE.paste(VGG_TREE.root, VGG16_TREE)
VGG_TREE.paste(VGG_TREE.root, VGG19_TREE)
