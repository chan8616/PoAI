import wx
from treelib import Tree
import gettext

# from model import MODELDICT
import model
from model import SVM_TREE, VGG_TREE, LOGISTIC_TREE, Xception_TREE
# from checkpoint import CHECKPOINTDICT
# from generator import DATASETDICT
from generator import IMAGE_CLASSIFICATION_TREE, IRIS_TREE
# from generator.image_classification import data_generator
from generator.image_classification import image_generator
from gui.frame import Frame

MODEL_TREE = Tree()
# MODEL_TREE.create_node('model', data=build)
# MODEL_TREE.create_node('model', data=train)
MODEL_TREE.create_node('model', data=model)
MODEL_TREE.paste(MODEL_TREE.root, SVM_TREE)
MODEL_TREE.paste(MODEL_TREE.root, VGG_TREE)
MODEL_TREE.paste(MODEL_TREE.root, LOGISTIC_TREE)
MODEL_TREE.paste(MODEL_TREE.root, Xception_TREE)


DATASET_TREE = Tree()
DATASET_TREE.create_node('generator', data=image_generator)
DATASET_TREE.paste(DATASET_TREE.root, IMAGE_CLASSIFICATION_TREE)
DATASET_TREE.paste(DATASET_TREE.root, IRIS_TREE)


class MyApp(wx.App):
    def OnInit(self):
        self.frame = Frame(DATASET_TREE, MODEL_TREE,
                           None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        self.frame.Maximize(True)
        return True


def main():
    gettext.install("app")

    app = MyApp(0)
    app.MainLoop()


if __name__ == '__main__':
    main()

