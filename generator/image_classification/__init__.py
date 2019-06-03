# import os
from pathlib import Path
from treelib import Tree
import sys
sys.path.insert(
    0, '/home/mlg/yys/project/TensorflowGUI/generator/image_classification')

from flow_from_directory import flow_from_directory_parser
from image_preprocess import image_preprocess_parser
from image_generator import image_generator_parser

# IMAGECLASSIFICATIONDICT = {
#     'mnist': mnist
# }

# MNIST_TREE = Tree()
# MNIST_TREE.create_node("mnist", data=mnist)

# FLOW_FROM_DIRECTORY_TREE = Tree()
# FLOW_FROM_DIRECTORY_TREE.create_node("flow_from_directory",
#                                      data=flow_from_directory_parser)
