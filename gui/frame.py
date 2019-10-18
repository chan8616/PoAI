import os
import sys
from types import ModuleType
from pathlib import Path
from gettext import gettext as _
from argparse import ArgumentParser
from pprint import pprint
from queue import Queue
from threading import Thread
# from multiprocessing import Process

import wx
from gooey import GooeyParser
from gooey.gui.containers.application import GooeyApplication as Page

# from wx.lib.agw.aui.auibook import AuiNotebook
#    from trees.datasettree import DatasetTree
#    from trees.modeltree import ModelTree

# from trees.dicttree import DictTree
# from notebook.notebook import Notebook
# from notebook.pages import Page
# from utils import Redirection

# from gui.trees.dicttree import DictTree
# from gui.trees.modeltree import ModelTree
# from gui.trees.datasettree import DatasetTree
#  from gui.tree_tree import TreeTree
from gui.module_tree import ModuleTree
from gui.notebook.notebook import Notebook
# from gui.notebook.pages import Page
from gui.utils import Redirection
from gui.progbar import TrainWindow, TrainThread

# if __name__ == '__main__':
#    from trees.datasettree import DatasetTree
#    from trees.modeltree import ModelTree
# #    from trees import DatasetTree, ModelTree
#    from notebook import Notebook
#    from utils import Redirection
# else:
#    from .trees.datasettree import DatasetTree
#    from .trees.modeltree import ModelTree
# #    from .trees import DatasetTree, ModelTree
#    from .notebook import Notebook
#    from .utils import Redirection

os.environ["UBUNTU_MENUPROXY"] = "0"


class Frame(wx.Frame):
    # begin wxGlade: MyFrame.__init__
    # def __init__(self, DATASETDICT, MODELDICT,
    #  def __init__(self, DATASET_TREE, MODEL_TREE,
    def __init__(self, dataset_module, model_module,
                 *args, **kwds):
        super(Frame, self).__init__(*args, **kwds)
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        # self.SetSize((1194, 922))
        # self.SetSize((1980, 450))

        self.cwd = os.getcwd()

        # Menu Bar
        self.menubar = wx.MenuBar()

        file_menu = wx.Menu()
        file_menu.Append(wx.ID_ANY, _("Import"), "")
        file_menu.Append(wx.ID_ANY, _("Export"), "")
        self.menubar.Append(file_menu, _("File"))

        datasets_menu = wx.Menu()
        datasets_menu.Append(wx.ID_ANY, _("Create"), "")
        datasets_menu.Append(wx.ID_ANY, _("Load"), "")
        self.menubar.Append(datasets_menu, _("Datasets"))

        models_menu = wx.Menu()
        models_menu.Append(wx.ID_ANY, _("Train Spec"), "")
        models_menu.Append(wx.ID_ANY, _("Test Spec"), "")
        # models_menu.Append(wx.ID_ANY, _("Run"), "")
        self.menubar.Append(models_menu, _("Models"))
        # self.SetMenuBar(self.menubar)
        # Menu Bar end

        # Tool Bar
        self.tool_bar = wx.ToolBar(self, wx.ID_ANY)
        self.SetToolBar(self.tool_bar)
        # self.tool_new = self.tool_bar.AddTool(
        #     1, _("New"), wx.Bitmap(
        #         "./icons/add.png", wx.BITMAP_TYPE_ANY),
        #     wx.NullBitmap, wx.ITEM_NORMAL, _("New"), "")
        # self.tool_load = self.tool_bar.AddTool(
        #     2, _("Load"), wx.Bitmap(
        #         "./icons/upload.png", wx.BITMAP_TYPE_ANY),
        #     wx.NullBitmap, wx.ITEM_NORMAL, _("Load"), "")
        # self.tool_save = self.tool_bar.AddTool(
        #     3, _("Save"), wx.Bitmap(
        #         "./icons/diskette(1).png", wx.BITMAP_TYPE_ANY),
        #     wx.NullBitmap, wx.ITEM_NORMAL, _("Save"), "")
        # self.tool_bar.AddSeparator()
        """
        self.tool_train_page = self.tool_bar.AddTool(
            4, _("Train Page"), wx.Bitmap(
                "gui/icons/3d-modeling.png", wx.BITMAP_TYPE_ANY),
            wx.NullBitmap, wx.ITEM_NORMAL, _("Train Spec"), "")

        self.tool_bar.AddSeparator()
        """

        self.tool_run = self.tool_bar.AddTool(
            5, _("Run"), wx.Bitmap(
                "gui/icons/play(1).png", wx.BITMAP_TYPE_ANY),
            wx.NullBitmap, wx.ITEM_NORMAL, _("Run"), "")
        self.tool_bar.EnableTool(self.tool_run.GetId(), False)

        """
        self.tool_bar.AddSeparator()

        self.tool_test_page = self.tool_bar.AddTool(
            6, _("Test Page"), wx.Bitmap(
                "gui/icons/background.png", wx.BITMAP_TYPE_ANY),
            wx.NullBitmap, wx.ITEM_NORMAL, _("Test"), "")
        """
        # Tool Bar end

        # TreeCtrl
        self.dataset_tree = self.tree_ctrl_1 = \
            ModuleTree(root_module=dataset_module, parent=self, id=wx.ID_ANY)
        #      TreeTree(tree=DATASET_TREE, parent=self, id=wx.ID_ANY)
        #      DatasetTree(DATASETDICT=DATASETDICT, parent=self, id=wx.ID_ANY)
        #      DictTree(trees, self, wx.ID_ANY)
        #      wx.TreeCtrl(self, wx.ID_ANY)
        self.model_tree = self.tree_ctrl_2 = \
            ModuleTree(root_module=model_module, parent=self, id=wx.ID_ANY)
        #      TreeTree(tree=MODEL_TREE, parent=self, id=wx.ID_ANY)
        #      ModelTree(MODELDICT=MODELDICT, parent=self, id=wx.ID_ANY)
        #      DictTree(trees, self, wx.ID_ANY)
        #      wx.TreeCtrl(self, wx.ID_ANY)
        self.item_to_page = dict()
        self.page_to_item = dict()
        # TreeCtrl end

        # Folders
        self.datset_folder = Path("dataset/").mkdir(exist_ok=True)
        self.checkpoint_folder = Path("checkpoint").mkdir(exist_ok=True)
        # Folders end

        # AuiNotebook
        self.notebook = self.notebook_1 = \
            Notebook(self, wx.ID_ANY)
        # wx.TextCtrl(self, wx.ID_ANY, "notebook",
        #             style=wx.HSCROLL | wx.TE_LEFT |
        #             wx.TE_MULTILINE | wx.TE_READONLY)
        # AuiNotebook(self, wx.ID_ANY)
        # MyNotebook(self, wx.ID_ANY)
        # AuiNotebook end

        # log window
        self.text_log = self.text_ctrl_1 = \
            wx.TextCtrl(self, wx.ID_ANY, "text_log",
                        style=wx.HSCROLL | wx.TE_LEFT |
                        wx.TE_MULTILINE | wx.TE_READONLY)
        self.redir = Redirection(self.text_log)
        #  sys.stdout = self.redir
        #  sys.stderr = self.redir
        # log window end

        self.__set_properties()
        self.__do_layout()
        self.__do_binds()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: MyFrame.__set_properties
        self.SetTitle(_("Tensorflow GUI"))
        self.tool_bar.Realize()
        self.text_ctrl_1.SetBackgroundColour(wx.Colour(235, 235, 235))
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: MyFrame.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        sizer_3.Add(self.tree_ctrl_1, 1, wx.EXPAND | wx.BOTTOM, 5)
        sizer_3.Add(self.tree_ctrl_2, 1, wx.EXPAND | wx.TOP, 5)
        sizer_2.Add(sizer_3, 2, wx.EXPAND, 0)
        sizer_2.Add(self.notebook_1, 20, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 2, wx.EXPAND, 0)
        sizer_1.Add(self.text_ctrl_1, 1, wx.ALL | wx.EXPAND, 0)
        # sizer_1.SetSizeHints(self)
        self.SetSizer(sizer_1)
        self.Layout()
        # end wxGlade

    def __do_binds(self):
        pass
        # tool bar
        """
        self.tool_bar.Bind(
            wx.EVT_TOOL, self.OnNew, id=self.tool_new.GetId())
        self.tool_bar.Bind(
            wx.EVT_TOOL, self.OnLoad, id=self.tool_load.GetId())
        self.tool_bar.Bind(
            wx.EVT_TOOL, self.OnSave, id=self.tool_save.GetId())
        self.tool_bar.Bind(
            wx.EVT_TOOL, self.OnTrainPage,
            id=self.tool_train_page.GetId())
        self.tool_bar.Bind(
            wx.EVT_TOOL, self.OnTestPage,
            id=self.tool_test_page.GetId())
        """
        self.tool_bar.Bind(
            wx.EVT_TOOL, self.OnRun,
            id=self.tool_run.GetId())

        # trees
        self.dataset_tree.Bind(
            wx.EVT_TREE_ITEM_ACTIVATED,
            self.datasetTreeOnActivated)
        self.model_tree.Bind(
            wx.EVT_TREE_ITEM_ACTIVATED,
            self.modelTreeOnActivated)
        """
        strainelf.dataset_tree.Bind(
            wx.EVT_TREE_ITEM_ACTIVATED, self.OnClosed)
        self.dataset_tree.Bind(
            wx.EVT_TREE_ITEM_EXPANDING, self.dataTreeOnExpand)
        self.model_tree.Bind(
            wx.EVT_TREE_ITEM_EXPANDING, self.modelTreeOnExpand)
        """

        # notebook
        self.notebook.Bind(
            wx.lib.agw.aui.auibook.EVT_AUINOTEBOOK_PAGE_CHANGED,
            self.OnPageChanged)

    def datasetTreeOnActivated(self, event):
        ItemID = event.GetItem()
        if ItemID in self.item_to_page:
            page, idx = self.notebook.FindTab(self.item_to_page[ItemID])
            if idx != 0:
                self.notebook.SetSelection(idx)
                return

        node = self.model_tree.GetItemData(ItemID)
        # ItemData = node.data
        if isinstance(node.data, ModuleType):
            title = _("Dataset Page")
            parser = node.data.Parser(GooeyParser())
        page = self.notebook.AddParserPage(
            parser, title)
        self.item_to_page[ItemID] = page
        self.page_to_item[page] = ItemID
        page.parser = parser

    def modelTreeOnActivated(self, event):
        ItemID = event.GetItem()
        if ItemID in self.item_to_page:
            page, idx = self.notebook.FindTab(self.item_to_page[ItemID])
            if idx != 0:
                self.notebook.SetSelection(idx)
                return

        model = self.model_tree.GetItemData(ItemID)

        build_parser = \
            model.build.build_parser(GooeyParser())
        build = model.build.build

        run_parser = \
            model.run.run_parser(GooeyParser())
        run = model.run.run

        generator_parser = \
            model.generator.generator_parser(GooeyParser())
        generator = model.generator.generator

        # model_parser.parse_args(['--help'])
        # dataset_parser.parse_args(['--help'])

        #  page = self.notebook.AddDoublePage(
        page = self.notebook.AddTriplePage(
            build_parser, run_parser, generator_parser,
            self.model_tree.GetItemText(ItemID))

        #  page = self.notebook.AddParserPage(
        #      build_parser, "Build Page")
        page.build_parser = build_parser
        page.build = build
        page.run_parser = run_parser
        page.run = run
        page.generator_parser = generator_parser
        page.generator = generator
        self.tool_bar.EnableTool(self.tool_run.GetId(), True)
        return

        node = self.model_tree.GetItemData(ItemID)
        ItemData = node.data
        print(node)
        if isinstance(ItemData, ModuleType):
            if hasattr(ItemData, 'build_parser'):
                title = _("Build Page")
                print('/'.join(ItemData.__name__.split('.')[1:]))
                print(node.identifier)
                parser = ItemData.build_parser(
                    # save_path=Path(
                    #     'checkpoint/' +
                    #     '/'.join(ItemData.__name__.split('.')[1:]) + '/')
                )

                # parser._defaults['save_path'] = \
                #     'checkpoint/' + \
                #     '/'.join(ItemData.__name__.split('.')[1:]) + '/'
                # parser._defaults['func'] = parser._defaults['build']
            elif hasattr(ItemData, 'main'):
                title = _("Test Page")
                parser = ItemData.main()
                page.main = parser._defaults['build']
            else:
                raise NotImplementedError
        elif isinstance(ItemData, Path):
            parentItemID = self.model_tree.GetItemParent(ItemID)
            parentItemData = self.model_tree.GetItemData(parentItemID)
            title = _("Train Page")
            parser = parentItemData.train_parser(
                GooeyParser(),
                load_path=ItemData)
            parser._defaults['func'] = parser._defaults['train']
        else:
            raise NotImplementedError
        page = self.notebook.AddParserPage(
            parser, title)
        # page = self.notebook.AddTrainPage(
        #     parser, parser, title)
        # self.item_to_page[ItemID] = page
        # self.page_to_item[page] = ItemID
        page.parser = parser
        self.tool_bar.EnableTool(self.tool_run.GetId(), True)

    def OnPageChanged(self, event):
        if self.notebook.isOnDoublePage():
            self.tool_bar.EnableTool(self.tool_run.GetId(), True)
        else:
            self.tool_bar.EnableTool(self.tool_run.GetId(), False)

        # self.dataset_tree.SelectItem(
        #     self.page_to_item[event.GetSelection()], True)
        # self.model_tree.SelectItem(
        #     self.page_to_item[event.GetSelection()], True)
        pass

    """
    def OnTrainPage(self, event):
        model_node = self.model_tree.GetItemData(
            self.model_tree.GetRootItem())
        dataset_node = self.dataset_tree.GetItemData(
            self.dataset_tree.GetRootItem())

        print(model_node.tag, model_node.data)
        train_setting_parser = \
            model_node.data.train.train_setting_parser(GooeyParser())
        train_setting = model_node.data.train.train_setting
        dataset_generator_parser = \
            dataset_node.data.image_generator_parser(GooeyParser())
        dataset_generator = dataset_node.data.image_generator
        print(train_setting_parser, dataset_generator_parser)

        # model_parser.parse_args(['--help'])
        # dataset_parser.parse_args(['--help'])

        page = self.notebook.AddTrainPage(
            train_setting_parser, dataset_generator_parser, "Train Page")
        page.train_setting_parser = train_setting_parser
        page.train_setting = train_setting

        page.dataset_generator_parser = dataset_generator_parser
        page.dataset_generator = dataset_generator

        page.run = model_node.data.train.train
        # page.test = model_node.data.test
        # page.model_parser = model_node.data.trainParser
        # page.dataset_parser = dataset_node.data.Parser
        self.tool_bar.EnableTool(self.tool_run.GetId(), True)
        # page
        pass
        # if hasattr(ItemData, 'trainParser'):
        #      title = _("Train Page")
        #      parser = ItemData.trainParser(GooeyParser())
        #  elif hasattr(ItemData, 'trainSettingParser'):
        #      title = _("Train Setting Page")
        #      parser = ItemData.trainSettingParser(GooeyParser())

    def OnTestPage(self, event):
        model_node = self.model_tree.GetItemData(
            self.model_tree.GetRootItem())
        dataset_node = self.dataset_tree.GetItemData(
            self.dataset_tree.GetRootItem())

        print(model_node.tag, model_node.data)
        test_setting_parser = \
            model_node.data.test.test_setting_parser(GooeyParser())
        test_setting = model_node.data.test.test_setting
        print(test_setting_parser)

        dataset_generator_parser = \
            dataset_node.data.image_generator_parser(GooeyParser())
        dataset_generator = \
            dataset_node.data.image_generator
        print(dataset_generator_parser)

        # model_parser.parse_args(['--help'])
        # dataset_parser.parse_args(['--help'])

        page = self.notebook.AddTestPage(
            test_setting_parser, dataset_generator_parser, "Test Page")
        page.test_setting_parser = test_setting_parser
        page.test_setting = test_setting

        page.dataset_generator_parser = dataset_generator_parser
        page.dataset_generator = dataset_generator

        page.run = model_node.data.test.test
        # page.model_parser = model_node.data.trainParser
        # page.dataset_parser = dataset_node.data.Parser
        self.tool_bar.EnableTool(self.tool_run.GetId(), True)

        # page
        pass
    """

    def OnRun(self, event):
        print("OnRun")
        page = self.notebook.GetPage(self.notebook.GetSelection())

        if self.notebook.isOnTriplePage():
            model_config = page.panel_1.navbar.getActiveConfig()
            model_config.resetErrors()
            run_config = page.panel_2.navbar.getActiveConfig()
            run_config.resetErrors()
            dataset_config = page.panel_3.navbar.getActiveConfig()
            dataset_config.resetErrors()

            if (model_config.isValid()
                    and run_config.isValid()
                    and dataset_config.isValid()):
                #  build_cmds = page.panel_1.buildCmd()
                #  build_cmds = page.panel_1.buildCliString()
                build_cmds = page.panel_1.buildString()
                #  run_cmds = page.panel_2.buildCmd()
                #  run_cmds = page.panel_2.buildCliString()
                run_cmds = page.panel_2.buildString()
                #  generator_cmds = page.panel_3.buildCmd()
                #  generator_cmds = page.panel_3.buildCliString()
                generator_cmds = page.panel_3.buildString()

                build_args = page.build_parser.parse_args(build_cmds)
                #  run_args = page.run_parser.parse_args(['train', '-h'])
                #  print(run_args)
                run_args = page.run_parser.parse_args(run_cmds)
                generator_args = page.generator_parser.parse_args(
                            generator_cmds)
                try:
                    #  generator_args = page.generator_parser.parse_args(
                    #          ['-h'])
                    #  generator_args = page.generator_parser.parse_args(
                    #          ['generator_balloon', '-h'])

                    self.train_with_progbar(page.run,
                                            (build_cmds, build_args,
                                             run_cmds, run_args,
                                             generator_cmds, generator_args))
                except Exception as e:
                    print(e)

        elif self.notebook.isOnDoublePage():
            model_config = page.panel_1.navbar.getActiveConfig()
            model_config.resetErrors()
            dataset_config = page.panel_2.navbar.getActiveConfig()
            dataset_config.resetErrors()

            if model_config.isValid() and dataset_config.isValid():
                cmds = page.panel_1.buildCmd()
                model_cmd, setting_cmds = cmds
                cmds = page.panel_2.buildCmd()
                dataset_cmd, dataset_generator_cmds = cmds
                print(model_cmd, setting_cmds)
                print(dataset_cmd, dataset_generator_cmds)

                # try:
                # print(page.train_setting_parser,
                #       page.dataset_generator_parser)
                if self.notebook.isOnTrainPage():
                    setting_parser = page.train_setting_parser
                    get_setting = page.train_setting
                elif self.notebook.isOnTestPage():
                    setting_parser = page.test_setting_parser
                    get_setting = page.test_setting
                else:
                    assert False, "page wierd"

                try:
                    setting_args = \
                        setting_parser.parse_args(
                            setting_cmds)
                    dataset_generator_args = \
                        page.dataset_generator_parser.parse_args(
                            dataset_generator_cmds)

                    pprint(setting_args)
                    pprint(dataset_generator_args)

                    setting = get_setting(model_cmd[0],
                                          setting_args)
                    print('setting', setting)
                    dataset_generator =\
                        page.dataset_generator(
                            dataset_cmd[0], dataset_generator_args)
                        #  page.dataset_generator_parser._defaults[
                        #      dataset_cmd[0]](dataset_generator_args)
                    print('dataset_generator', dataset_generator)
                    # page.train(train_setting, dataset_generator)
                    page.run(model_cmd[0], setting, dataset_generator)
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise NotImplementedError
                
                # args = page.parser.parse_args()
                # args.func(args)
                # except:
                #     print("train load error occurs foo")
            else:
                model_config.displayErrors()
                dataset_config.displayErrors()
                page.Layout()

        else:
            config = page.navbar.getActiveConfig()
            config.resetErrors()
            if config.isValid():
                cmd, cmds = page.buildCmd()
                print('cmds:', cmds)

                parser = page.build_parser
                # print(parser)
                # # args = parser.parse_args(positional+optional)
                args = parser.parse_args(cmds)
                if cmd == []:
                    page.build(args)
                else:
                    page.build(cmd[0], args)

                self.model_tree.DeleteChildren(self.model_tree.root_id)
                self.model_tree.ExtendTree(self.model_tree.root_id)
                self.model_tree.Expand(self.model_tree.root_id)
            else:
                config.displayErrors()
                page.Layout()

        # page
        pass

    def train_with_progbar(self, train_function, config):
        stream = Queue()
        window = TrainWindow(self, title='Train Progress', stream=stream)
        train_thread = TrainThread(train_function, config, stream)
        progbar_thread = Thread(target=window.main_loop)

        train_thread.start()
        progbar_thread.start()





if __name__ == '__main__':
    # from argparse import ArgumentParser
    app = wx.App(0)
    test_dict = {
        'k1': 'v1',
        'd1': {
            'k2': 'v2',
            'd2': {
                'k3': 'v3',
            },
        }
    }
    parser1 = ArgumentParser()
    parser1.add_argument("test1")

    parser2 = ArgumentParser()
    parser2.add_argument("test2")

    datasetDict = test_dict
    modelDict = test_dict

    frame = Frame(datasetDict, modelDict, None)

#    frame.notebook.AddPage(
#        Page(parser1, frame, wx.ID_ANY),
#        "test")
    page1 = frame.notebook.AddPage(
        Page(parser1, frame, wx.ID_ANY),
        _("Page 1"),
        select=True)
    page2 = frame.notebook.AddPage(
        Page(parser2, frame, wx.ID_ANY),
        _("Page 1"),
        select=True)

#    app.SetTopWindow(frame)
    frame.Show()
    app.MainLoop()
