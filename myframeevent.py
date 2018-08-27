import wx
#import wx.lib.inspection
import os

from myframe import MyFrame

import sys

from utils.util import Redirection

class MyFrameEvent(MyFrame):
    def __init__(self, *args, **kwds):
        super(MyFrameEvent, self).__init__(*args, **kwds)
        self.cwd = os.getcwd()

        # tool bar
        self.tool_bar.Bind(wx.EVT_TOOL, self.OnNew, id=self.tool_new.GetId())
        self.tool_bar.Bind(wx.EVT_TOOL, self.OnLoad, id=self.tool_load.GetId())
        self.tool_bar.Bind(wx.EVT_TOOL, self.OnSave, id=self.tool_save.GetId())
        self.tool_bar.Bind(wx.EVT_TOOL, self.OnTrainSpec, id=self.tool_train_spec.GetId())
        self.tool_bar.Bind(wx.EVT_TOOL, self.OnTrainStart, id=self.tool_train_start.GetId())
        self.tool_bar.Bind(wx.EVT_TOOL, self.OnTest, id=self.tool_test.GetId())

        # trees
        self.data_tree = self.tree_ctrl_1
        self.model_tree = self.tree_ctrl_2
        self.item_to_page = dict()

        # load dataset tree
        self.datasetDir = os.path.join(self.cwd, "Dataset")
        if not os.path.exists(self.datasetDir):
            os.makedirs(self.datasetDir)
        self.buildTree(self.data_tree, self.datasetDir)
        self.data_tree.Expand(self.data_tree.GetRootItem())
       
        # load model tree (in Modules folder)
        self.modelDir= os.path.join(self.cwd, "Modules")
        if not os.path.exists(self.modelDir):
            os.makedirs(self.modelDir)
        self.buildTree(self.model_tree, self.modelDir)
        self.model_tree.Expand(self.model_tree.GetRootItem())

        # load pretrained model tree (in checkpoint folder)
        self.pretrainedModelDir= os.path.join(self.cwd, "checkpoint")
        if os.path.exists(self.pretrainedModelDir):
            for pretrainedModel in os.listdir(self.pretrainedModelDir):
                pretrainedModelPath = os.path.join(self.pretrainedModelDir, pretrainedModel)
                self.buildTree(self.model_tree, pretrainedModelPath)

        self.data_tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.dataTreeOnActivated)
        self.model_tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.modelTreeOnActivated)

        # notebook
        self.notebook = self.notebook_1
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSED, self.OnClosed)
        #self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.OnPageChanged)
        self.notebook.Bind(wx.lib.agw.aui.auibook.EVT_AUINOTEBOOK_PAGE_CHANGED, self.OnPageChanged)
        # panels
        self.dataSpec_panel_list = self.notebook_1.panel_1_list
        #self.createDataSpecPanel(self.notebook_1, dict())
        self.modelSpec_panel_list = self.notebook_1.panel_2_list
        #self.createModelSpecPanel(self.notebook_1, dict())
        self.trainSpec_panel_list = self.notebook_1.panel_3_list
        #self.createTrainSpecPanel(self.notebook_1, dict())

        # train spec default
        self.train_spec = {'max_iter': 10000, 'lr':1e-3, 'optimizer':'Adam', 'seed':0, 'batch_size':32, 'checkpoint interval':1000, 'validation interval':1000}

        # text log
        self.text_log = self.text_ctrl_1
        self.redir = Redirection(self.text_log)
        sys.stdout = self.redir

    def OnClose(self, event):
        print('close')
    def OnClosed(self, event):
        print('closed')
    def OnPageChanged(self, event):
        #print('page changed', event, event.GetNotifyEvent())
        
        idx = self.notebook.GetSelection()
        page = self.notebook.GetPage(idx) 
        print(page)

    def OnToolBar(self, event):
        print(event)
        print(event.GetInt())
    def OnNew(self, event):
        pass
    def OnLoad(self, event):
        pass
    def OnSave(self, event):
        pass
    def OnTest(self, event):
        pass
    def OnTrainSpec(self, event):
        self.createTrainSpecPanel(self.notebook_1, self.train_spec)
    def OnTrainStart(self, event):
        pass

    def OnDataSpec(self, item):
        dict = self.setDataSpec(item)
        page = self.createDataSpecPanel(self.notebook_1, dict)
        self.item_to_page[item] = page

    def OnModelSpec(self, item):
        dict = self.setModelSpec(item)
        page = self.createModelSpecPanel(self.notebook_1, dict)
        self.item_to_page[item] = page

    def treeOnActivated(self, tree, OnSpecFun):
        item = tree.GetFocusedItem()
        if tree.GetItemParent(item) == \
                tree.GetRootItem():

            if item not in self.item_to_page:
                OnSpecFun(item)
            else:
                _, idx = self.notebook.FindTab(self.item_to_page[item])
                if idx == -1:
                    OnSpecFun(item)
                else:
                    self.notebook.SetSelection(idx)

    def dataTreeOnActivated(self, event): 
        self.treeOnActivated(self.data_tree, self.OnDataSpec)

    def modelTreeOnActivated(self, event):
        self.treeOnActivated(self.model_tree, self.OnModelSpec)

    def setDataSpec(self, dataID):
        res = dict()
        name = self.data_tree.GetItemText(dataID)
        res['name'] = name
        path = self.data_tree.GetItemData(dataID)
        res['path'] = path


        return res

    def setModelSpec(self, modelID):
        res = dict()
        return res
        

    def buildTree(self, tree, rootDir, treeRoot=None):
        if treeRoot is None: treeRoot = tree.GetRootItem()

        def itemExist(tree, data, rootID):
            item, cookie = tree.GetFirstChild(rootID)
            while item.IsOk():
                if tree.GetItemData(item) == data:
                    return True
                item, cookie = tree.GetNextChild(rootID, cookie)
            return False

        if tree.IsEmpty() or not itemExist(tree, rootDir, treeRoot):
            rootID = tree.AppendItem(treeRoot, (os.path.basename(rootDir)))
            tree.SetItemData(rootID, rootDir)
            self.extendTree(tree, rootID)
        else:
            print("Dataset is already exist!")

    def extendTree(self, tree, parentID):
        parentPath = tree.GetItemData(parentID)

        subdirs = os.listdir(parentPath)
        subdirs.sort()
        for child in subdirs:
            childPath = os.path.join(parentPath, child)
            if os.path.isdir(childPath) and not os.path.islink(child):
                childID = tree.AppendItem(parentID, child)
                tree.SetItemData(childID, childPath)

                grandsubdirs = os.listdir(childPath)
                grandsubdirs.sort()
                for grandchild in grandsubdirs:
                    grandchildPath = os.path.join(childPath, grandchild)
                    if os.path.isdir(grandchildPath) and not os.path.islink(grandchildPath):
                        grandchildID = tree.AppendItem(childID, grandchild)
                        tree.SetItemData(grandchildID, grandchildPath)



