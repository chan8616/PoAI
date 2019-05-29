import os
from pathlib import Path
from treelib import Tree
import sys
sys.path.insert(0, '/home/mlg/yys/project/TensorflowGUI/model/vgg')

from vgg.build import build_parser

# from vgg16 import Parser
# from model import vgg
# from model.vgg import vgg16
# from model.vgg import vgg19
# from model.vgg.train import train
# VGGDICT = {
#     'vgg16': vgg16,
#     'vgg19': vgg19
# }

from vgg import vgg16
from vgg import vgg19
VGG16_TREE = Tree()
VGG16_TREE.create_node(
    'vgg16',
    # 'model/vgg/vgg16/',
    Path(vgg16.__path__[0]).relative_to(Path(os.getcwd())),
    data=vgg16)
VGG19_TREE = Tree()
VGG19_TREE.create_node(
    'vgg19',
    Path(vgg19.__path__[0]).relative_to(Path(os.getcwd())),
    data=vgg19)
