import wx
from gooey import GooeyParser
from gettext import gettext as _

from wx.lib.agw.aui.auibook import AuiNotebook

from gui.notebook.build_spec_from_parser import build_spec_from_parser
from gui.notebook.page import Page, DoublePage, TriplePage


class Notebook(AuiNotebook):
    def __init__(self, *args, **kwds):
        # Auinotebook
        super(Notebook, self).__init__(*args, **kwds)

        self.num_page = 0
        self.num_dataset_page = 0
        self.num_model_page = 0
        self.num_train_page = 0
        self.num_test_page = 0

        self.__do_layout()
        # Auinotebook end

    def __do_layout(self):
        self.Layout()

    def AddParserPage(self, parser, caption, *args, **kwds):
        build_spec = build_spec_from_parser(
            parser, **kwds)
        # pprint(build_spec)
        # page = SinglePage(parser, self, *args, **kwds)
        page = Page(build_spec, self)
        self.AddPage(page, caption, select=True, *args, **kwds)
        return page

    def AddDoublePage(self,
                      build_parser: GooeyParser,
                      dataset_parser: GooeyParser,
                      caption: str,
                      *args, **kwds):
        double_page = DoublePage(
            build_parser,
            dataset_parser,
            'Selecet Build Mode',
            'Select Dataset Format',
            self, *args, **kwds)
        self.AddPage(double_page, caption, select=True, *args, **kwds)
        return double_page

    def AddTriplePage(self,
                      build_parser: GooeyParser,
                      run_parser: GooeyParser,
                      dataset_parser: GooeyParser,
                      caption: str,
                      *args, **kwds):
        triple_page = TriplePage(
            build_parser,
            run_parser,
            dataset_parser,
            'Selecet Build Mode',
            'Selecet Run Mode',
            'Select Dataset Format',
            self, *args, **kwds)
        self.AddPage(triple_page, caption, select=True, *args, **kwds)
        return triple_page

    def AddTrainPage(self, model_parser, dataset_parser, caption,
                     *args, **kwds):
        train_page = DoublePage(
            model_parser,
            dataset_parser,
            'Selecet Model',
            'Select Dataset Format',
            self, *args, **kwds)
        train_page.phase = "train"
        self.AddPage(train_page, caption, select=True, *args, **kwds)
        return train_page

    def AddTestPage(self, model_parser, dataset_parser, caption,
                    *args, **kwds):
        test_page = DoublePage(
            model_parser,
            dataset_parser,
            'Selecet Model',
            'Select Dataset Format',
            self, *args, **kwds)
        test_page.phase = "test"
        self.AddPage(test_page, caption, select=True, *args, **kwds)
        return test_page

    def isOnTrainPage(self):
        page = self.GetPage(self.GetSelection())
        return True if self.isOnDoublePage and page.phase == 'train' else False

    def isOnTestPage(self):
        page = self.GetPage(self.GetSelection())
        return True if self.isOnDoublePage and page.phase == 'test' else False

    def isOnDoublePage(self):
        page = self.GetPage(self.GetSelection())
        return isinstance(page, DoublePage)

    def isOnTriplePage(self):
        page = self.GetPage(self.GetSelection())
        return isinstance(page, TriplePage)


if __name__ == '__main__':
    # parser = ArgumentParser()
    # #parser.add_argument("header", default="TrainSpec")
    # parser.add_argument(
    #    "model", type=str, 
    #    choices=["logistic/simple-logistic",
    #             "logistic/multilayer-logistic",
    #             "svm/svc",
    #             "vgg/vgg19"])

    # parser.add_argument(
    #    "load", type=str,
    #    choices=["simple-logistic/mnist_1",
    #             "simple-logistic/cifar10_1"])
    # parser.add_argument(
    #    "dataset", type=str,
    #    choices=["mnist",
    #             "cifar10"])

    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--gpu", action='store_true')
    # parser.add_argument("--optimizer")
    # parser.add_argument("--learning_rate")
    # parser.add_argument("--validation_step")

    from argparse import ArgumentParser
    parser1 = ArgumentParser()
    parser1.add_argument("test1")

    parser2 = ArgumentParser()
    parser2.add_argument("test2")

    app = wx.App()
    frame = NotebookFrame(None)
    page1 = frame.AddPage(
        Page(parser1, frame, wx.ID_ANY))
    page2 = frame.AddPage(
        Page(parser2, frame, wx.ID_ANY))


#    def AddPage(self, parser, *args, **kwds):
#        page = Page(parser, *args, **kwds)
#        self.AddPage(
#            page, 
#            _("Page %d"%self.num_page), 
#            select=True)
#        return page
#    page = frame.notebook.AddPage(parser1, frame, wx.ID_ANY)
#    page = frame.notebook.AddPage(parser2, frame, wx.ID_ANY)
    frame.Show()
    app.MainLoop()

