import os
from pathlib import Path
from treelib import Tree
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'model'))

from model import build, train, test

import svm

import logistic
from logistic import SIMPLE_LOGISTIC_TREE
from logistic import MULTILAYER_LOGISTIC_TREE

import vgg
from vgg import VGG16_TREE, VGG19_TREE

import Xception
import MobileNet

SVM_TREE = Tree()
SVM_TREE.create_node(
    'svm',
    Path(svm.__path__[0]).relative_to(os.getcwd()),
    data=svm)

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

Xception_TREE = Tree()
Xception_TREE.create_node(
    'Xception',
    Path(Xception.__path__[0]).relative_to(os.getcwd()),
    data=Xception)

MobileNet_TREE = Tree()
MobileNet_TREE.create_node(
    'MobileNet',
    Path(MobileNet.__path__[0]).relative_to(os.getcwd()),
    data=MobileNet)
