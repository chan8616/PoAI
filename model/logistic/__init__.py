import os
from pathlib import Path
from treelib import Tree
import sys
sys.path.insert(0, '/home/mlg/yys/project/TensorflowGUI/model/logistic')

from logistic.build import build_parser
from logistic import simple_logistic
from logistic import multilayer_logistic
SIMPLE_LOGISTIC_TREE = Tree()
SIMPLE_LOGISTIC_TREE.create_node(
    'simple_logistic',
    # 'model/vgg/vgg16/',
    Path(simple_logistic.__path__[0]).relative_to(Path(os.getcwd())),
    data=simple_logistic)
MULTILAYER_LOGISTIC_TREE = Tree()
MULTILAYER_LOGISTIC_TREE.create_node(
    'multilayer_logistic',
    Path(multilayer_logistic.__path__[0]).relative_to(Path(os.getcwd())),
    data=multilayer_logistic)
