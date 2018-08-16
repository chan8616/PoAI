# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.8.2 on Thu Aug  9 10:26:51 2018
#

import wx
# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade
import os
from MyDialog import MyDialog

# import dircache # dircache is not supported on python 3.x

import sys

VERSION_2 = True if sys.version[0] == '2' else False

if VERSION_2:
    import dircache

from utils.util import Redirection

wildcard = "Python source (*.py)|*.py|" \
            "All files (*.*)|*.*"

class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyFrame.__init__
        wx.Frame.__init__(self, *args, **kwds)
        self.currentDirectory = os.getcwd()
        self.main_tab = wx.Notebook(self, wx.ID_ANY, style=0)
        self.data = wx.Panel(self.main_tab, wx.ID_ANY)
        self.data_left = wx.Panel(self.data, wx.ID_ANY, style=wx.BORDER_SUNKEN)
        self.data_list = wx.Panel(self.data_left, wx.ID_ANY)
        #self.data_dir = wx.GenericDirCtrl(self.data_left, -1, dir=self.currentDirectory)
        self.data_tree = wx.TreeCtrl(self.data_left, wx.ID_ANY)#, style=wx.TR_HIDE_ROOT)
        self.data_tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.data_tree_OnActivated)
        #self.data_tree.AddRoot("Registered Datasets")
        self.buildTree(self.data_tree, os.path.join(self.currentDirectory, "dataset"))
        self.data_tree.Expand(self.data_tree.GetRootItem())
        self.data_new = wx.Button(self.data_left, wx.ID_ANY, _("New"))
        #self.data_new.Bind(wx.EVT_BUTTON, self.onDir)
        self.data_load_button = wx.Button(self.data_left, wx.ID_ANY, _("Load"))
        self.data_load_button.Bind(wx.EVT_BUTTON, self.data_load_button_clicked)  #self.onDir)
        self.data_save = wx.Button(self.data_left, wx.ID_ANY, _("Save"))
        self.data_right = wx.Notebook(self.data, wx.ID_ANY, style=0)
        self.data_spec = wx.Panel(self.data_right, wx.ID_ANY)
        self.text_ctrl_9 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_9", style=wx.TE_READONLY)
        self.text_ctrl_10 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_10", style=wx.TE_READONLY)
        self.text_ctrl_11 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_11", style=wx.TE_READONLY)
        self.text_ctrl_12 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_12")
        self.text_ctrl_13 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_13")
        self.combo_box_2 = wx.ComboBox(self.data_spec, wx.ID_ANY, choices=[], style=wx.CB_DROPDOWN)
        self.text_ctrl_15 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_15")
        self.text_ctrl_16 = wx.TextCtrl(self.data_spec, wx.ID_ANY, "text_ctrl_16")
        self.data_select = wx.Button(self.data_spec, wx.ID_ANY, _("Select"))
        self.data_select.Bind(wx.EVT_BUTTON, self.data_select_button_clicked)  #self.onDir)
        #self.data_log = wx.TextCtrl(self.data, wx.ID_ANY, _("data_log\n"), style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.models = wx.Panel(self.main_tab, wx.ID_ANY)
        self.model_left = wx.Panel(self.models, wx.ID_ANY, style=wx.BORDER_SUNKEN)
        #self.model_list = wx.Panel(self.model_left, wx.ID_ANY)
        #self.model_dir = wx.GenericDirCtrl(self.model_left, -1, dir=self.currentDirectory)
        self.model_tree = wx.TreeCtrl(self.model_left, wx.ID_ANY)#, style=wx.TR_HIDE_ROOT)
        #self.model_tree.AddRoot("Pretrained Models")
        self.buildTree(self.model_tree, os.path.join(os.getcwd(), "model"))
        self.model_tree.Expand(self.model_tree.GetRootItem())
        self.model_make_button = wx.Button(self.model_left, wx.ID_ANY, _("Make Pretrained Model"))
        self.model_make_button.Bind(wx.EVT_BUTTON, self.model_make_button_clicked) # = wx.Button(self.model_left, wx.ID_ANY, _("Make Pretrained Model"))
        self.model_right = wx.Notebook(self.models, wx.ID_ANY, style=0)
        self.model_test_single = wx.Panel(self.model_right, wx.ID_ANY)
        self.button_1 = wx.Button(self.model_test_single, wx.ID_ANY, _("Browse"))
        self.button_1.Bind(wx.EVT_BUTTON, self.onOpenFile)
        self.text_ctrl_1 = wx.TextCtrl(self.model_test_single, wx.ID_ANY, "text_ctrl_1", style=wx.TE_READONLY)
        self.button_7 = wx.Button(self.model_test_single, wx.ID_ANY, _("Classify One"))
        self.model_test_folder = wx.Panel(self.model_right, wx.ID_ANY)
        self.button_4 = wx.Button(self.model_test_folder, wx.ID_ANY, _("Browse"))
        self.button_4.Bind(wx.EVT_BUTTON, self.onDir)
        self.text_ctrl_2 = wx.TextCtrl(self.model_test_folder, wx.ID_ANY, "text_ctrl_2", style=wx.TE_READONLY)
        self.text_ctrl_3 = wx.TextCtrl(self.model_test_folder, wx.ID_ANY, "text_ctrl_3")
        self.text_ctrl_4 = wx.TextCtrl(self.model_test_folder, wx.ID_ANY, "text_ctrl_4")
        self.button_6 = wx.Button(self.model_test_folder, wx.ID_ANY, _("Classify Many"))
        self.button_5 = wx.Button(self.model_test_folder, wx.ID_ANY, _("Top N predictions per Category"))

        self.model_pretrained_data = wx.Panel(self.model_right, wx.ID_ANY)
        self.model_pretrained_data_list = wx.ListBox(self.model_pretrained_data, wx.ID_ANY, choices=[_("MNIST"), _("CIFAR-10"), _("CIFAR-100")], style=wx.LB_ALWAYS_SB | wx.LB_SINGLE)
        self.panel_1 = wx.Panel(self.model_pretrained_data, wx.ID_ANY)
        self.model_pretrained_option = wx.Panel(self.model_right, wx.ID_ANY)
        self.text_ctrl_5 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_5")
        self.text_ctrl_17 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_17")
        self.text_ctrl_6 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_6")
        self.text_ctrl_18 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_18")
        self.text_ctrl_7 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_7")
        self.text_ctrl_19 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_19")
        self.text_ctrl_8 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_8")
        self.text_ctrl_20 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_20", style=wx.TE_MULTILINE)
        self.text_ctrl_14 = wx.TextCtrl(self.model_pretrained_option, wx.ID_ANY, "text_ctrl_14")
        self.button_10 = wx.Button(self.model_pretrained_option, wx.ID_ANY, _("Advanced Options"))
        self.button_11 = wx.Button(self.model_pretrained_option, wx.ID_ANY, _("Train"))

        #self.model_log = wx.TextCtrl(self.models, wx.ID_ANY, "", style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.log = wx.TextCtrl(self, wx.ID_ANY, _("log\n"), style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

        # Redirect stdout to data_log
        #self.redir = Redirection(self.data_log)
        self.redir = Redirection(self.log)
        sys.stdout = self.redir


    def __set_properties(self):
        # begin wxGlade: MyFrame.__set_properties
        self.SetTitle(_("Tensorflow GUI"))
        #self.data_log.SetBackgroundColour(wx.Colour(235, 235, 235))
        self.model_pretrained_data_list.SetSelection(0)
        #self.model_log.SetBackgroundColour(wx.Colour(235, 235, 235))
        self.log.SetBackgroundColour(wx.Colour(235, 235, 235))
        # end wxGlade


        # Create, Save delete
        self.data_new.SetBackgroundColour(wx.Colour(235, 235, 235))
        self.data_save.SetBackgroundColour(wx.Colour(235, 235, 235))

    def __do_layout(self):
        # begin wxGlade: MyFrame.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_8 = wx.BoxSizer(wx.VERTICAL)
        sizer_9 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_10 = wx.BoxSizer(wx.HORIZONTAL)
        grid_sizer_4 = wx.GridBagSizer(0, 0)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        grid_sizer_3 = wx.GridBagSizer(0, 0)
        sizer_12 = wx.BoxSizer(wx.HORIZONTAL)
        grid_sizer_1 = wx.GridBagSizer(0, 0)
        sizer_11 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        grid_sizer_2 = wx.GridBagSizer(0, 0)
        sizer_4 = wx.BoxSizer(wx.VERTICAL)
        sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        #sizer_4.Add(self.data_list, 10, 0, 0)
        #sizer_4.Add(self.data_dir, 10, wx.ALL | wx.EXPAND, 0)
        sizer_4.Add(self.data_tree, 10, wx.ALL | wx.EXPAND, 0)
        sizer_6.Add(self.data_new, 1, wx.ALIGN_CENTER, 0)
        sizer_6.Add(self.data_load_button, 1, wx.ALIGN_CENTER, 0)
        sizer_6.Add(self.data_save, 1, wx.ALIGN_CENTER, 0)
        sizer_4.Add(sizer_6, 1, wx.ALIGN_CENTER, 0)
        self.data_left.SetSizer(sizer_4)
        sizer_3.Add(self.data_left, 1, wx.EXPAND, 0)
        label_9 = wx.StaticText(self.data_spec, wx.ID_ANY, _("Dataset Path"))
        label_9.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_9, (1, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        grid_sizer_2.Add(self.text_ctrl_9, (1, 7), (1, 27), wx.EXPAND, 0)
        label_10 = wx.StaticText(self.data_spec, wx.ID_ANY, _("Num of Classes"))
        label_10.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_10, (2, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        grid_sizer_2.Add(self.text_ctrl_10, (2, 7), (1, 27), wx.EXPAND, 0)
        label_11 = wx.StaticText(self.data_spec, wx.ID_ANY, _("Num of Images"))
        label_11.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_11, (3, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        grid_sizer_2.Add(self.text_ctrl_11, (3, 7), (1, 27), wx.EXPAND, 0)
        label_12 = wx.StaticText(self.data_spec, wx.ID_ANY, _("Image Size"))
        label_12.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_12, (4, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        grid_sizer_2.Add(self.text_ctrl_12, (4, 7), (1, 13), wx.EXPAND, 0)
        label_13 = wx.StaticText(self.data_spec, wx.ID_ANY, _("X"))
        grid_sizer_2.Add(label_13, (4, 20), (1, 1), wx.ALIGN_CENTER, 0)
        grid_sizer_2.Add(self.text_ctrl_13, (4, 21), (1, 13), wx.EXPAND, 0)
        label_14 = wx.StaticText(self.data_spec, wx.ID_ANY, _("Image Type"))
        label_14.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_14, (5, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        grid_sizer_2.Add(self.combo_box_2, (5, 7), (1, 27), wx.EXPAND, 0)
        label_15 = wx.StaticText(self.data_spec, wx.ID_ANY, _("% of testing"))
        label_15.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_15, (6, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        grid_sizer_2.Add(self.text_ctrl_15, (6, 7), (1, 27), wx.EXPAND, 0)
        label_16 = wx.StaticText(self.data_spec, wx.ID_ANY, _("Maximum samples per class"))
        label_16.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_16, (7, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        grid_sizer_2.Add(self.text_ctrl_16, (7, 7), (1, 27), wx.EXPAND, 0)
        grid_sizer_2.Add(self.data_select, (18, 35), (1, 4), 0, 0)
        self.data_spec.SetSizer(grid_sizer_2)
        grid_sizer_2.AddGrowableRow(17)
        grid_sizer_2.AddGrowableCol(7)
        grid_sizer_2.AddGrowableCol(21)
        self.data_right.AddPage(self.data_spec, _("Specification"))
        sizer_5.Add(self.data_right, 1, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_3.Add(sizer_5, 3, wx.EXPAND, 0)
        sizer_2.Add(sizer_3, 2, wx.EXPAND, 0)
        #sizer_2.Add(self.data_log, 1, wx.EXPAND, 0)
        self.data.SetSizer(sizer_2)
        #sizer_11.Add(self.model_list, 10, 0, 0)
        #sizer_11.Add(self.model_dir, 10, wx.ALL | wx.EXPAND, 0)
        sizer_11.Add(self.model_tree, 10, wx.ALL | wx.EXPAND, 0)
        sizer_11.Add(self.model_make_button, 0, wx.ALIGN_CENTER, 0)
        self.model_left.SetSizer(sizer_11)
        sizer_9.Add(self.model_left, 1, wx.EXPAND, 0)
        label_1 = wx.StaticText(self.model_test_single, wx.ID_ANY, _("Upload image"))
        label_1.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, ""))
        grid_sizer_1.Add(label_1, (1, 1), (1, 5), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.button_1, (3, 1), (1, 2), wx.EXPAND, 0)
        grid_sizer_1.Add(self.text_ctrl_1, (3, 3), (1, 24), wx.EXPAND, 0)
        grid_sizer_1.Add(self.button_7, (17, 27), (1, 1), 0, 0)
        grid_sizer_1.AddGrowableRow(15)
        grid_sizer_1.AddGrowableRow(16)
        grid_sizer_1.AddGrowableCol(24)
        grid_sizer_1.AddGrowableCol(25)
        grid_sizer_1.AddGrowableCol(26)
        sizer_12.Add(grid_sizer_1, 1, wx.EXPAND, 0)
        self.model_test_single.SetSizer(sizer_12)
        label_2 = wx.StaticText(self.model_test_folder, wx.ID_ANY, _("Upload image folder"))
        label_2.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, ""))
        grid_sizer_3.Add(label_2, (1, 1), (1, 4), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_3.Add(self.button_4, (3, 1), (1, 2), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_2, (3, 3), (1, 22), wx.EXPAND, 0)
        label_3 = wx.StaticText(self.model_test_folder, wx.ID_ANY, _("Number of images use from the file"))
        label_3.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, ""))
        grid_sizer_3.Add(label_3, (5, 1), (1, 8), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_3, (6, 1), (1, 24), wx.EXPAND, 0)
        label_4 = wx.StaticText(self.model_test_folder, wx.ID_ANY, _("Leave blank to use all"))
        label_4.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.LIGHT, 0, ""))
        grid_sizer_3.Add(label_4, (7, 1), (1, 4), 0, 0)
        label_5 = wx.StaticText(self.model_test_folder, wx.ID_ANY, _("Number of images to show per category"))
        label_5.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_5, (9, 1), (1, 8), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_4, (10, 1), (1, 24), wx.EXPAND, 0)
        grid_sizer_3.Add(self.button_6, (18, 24), (1, 3), 0, 0)
        grid_sizer_3.Add(self.button_5, (18, 27), (1, 2), 0, 0)
        self.model_test_folder.SetSizer(grid_sizer_3)
        grid_sizer_3.AddGrowableRow(14)
        grid_sizer_3.AddGrowableRow(15)
        grid_sizer_3.AddGrowableCol(16)
        grid_sizer_3.AddGrowableCol(17)

        sizer_7.Add(self.model_pretrained_data_list, 1, wx.ALL | wx.EXPAND, 0)
        sizer_7.Add(self.panel_1, 2, wx.EXPAND, 0)
        self.model_pretrained_data.SetSizer(sizer_7)
        label_6 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Training epochs"))
        label_6.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_6, (1, 1), (1, 10), 0, 0)
        label_19 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Batch size"))
        label_19.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_19, (1, 20), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_5, (2, 1), (1, 17), wx.EXPAND, 0)
        grid_sizer_4.Add(self.text_ctrl_17, (2, 20), (1, 17), wx.EXPAND, 0)
        label_7 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Snapshot interval (in epochs)"))
        label_7.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_7, (4, 1), (1, 12), 0, 0)
        label_20 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Solver type"))
        label_20.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_20, (4, 20), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_6, (5, 1), (1, 17), wx.EXPAND, 0)
        grid_sizer_4.Add(self.text_ctrl_18, (5, 20), (1, 17), wx.EXPAND, 0)
        label_8 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Validation interval (in epochs)"))
        label_8.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_8, (7, 1), (1, 12), 0, 0)
        label_21 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Base learning rate"))
        label_21.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_21, (7, 20), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_7, (8, 1), (1, 17), wx.EXPAND, 0)
        grid_sizer_4.Add(self.text_ctrl_19, (8, 20), (1, 17), wx.EXPAND, 0)
        label_17 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Random seed"))
        label_17.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_17, (10, 1), (1, 10), 0, 0)
        label_22 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Select which GPU[s] you would like to use"))
        label_22.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_22, (10, 20), (1, 16), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_8, (11, 1), (1, 17), wx.EXPAND, 0)
        grid_sizer_4.Add(self.text_ctrl_20, (11, 20), (4, 17), wx.EXPAND, 0)
        label_18 = wx.StaticText(self.model_pretrained_option, wx.ID_ANY, _("Model name"))
        label_18.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_18, (13, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_14, (14, 1), (1, 17), wx.EXPAND, 0)
        grid_sizer_4.Add(self.button_10, (18, 36), (1, 2), 0, 0)
        grid_sizer_4.Add(self.button_11, (18, 38), (1, 1), 0, 0)
        self.model_pretrained_option.SetSizer(grid_sizer_4)
        grid_sizer_4.AddGrowableRow(17)
        grid_sizer_4.AddGrowableCol(17)
        grid_sizer_4.AddGrowableCol(35)
        self.model_right.AddPage(self.model_test_single, _("Test (Single Image)"))
        self.model_right.AddPage(self.model_test_folder, _("Test (Image Folder)"))
        self.model_right.AddPage(self.model_pretrained_data, _("Select Dataset"))
        self.model_right.AddPage(self.model_pretrained_option, _("Options"))
        sizer_10.Add(self.model_right, 1, wx.ALIGN_CENTER | wx.EXPAND, 0)
        sizer_9.Add(sizer_10, 3, wx.EXPAND, 0)
        sizer_8.Add(sizer_9, 2, wx.EXPAND, 0)
        #sizer_8.Add(self.model_log, 1, wx.EXPAND, 0)
        self.models.SetSizer(sizer_8)
        self.main_tab.AddPage(self.data, _("Datasets"))
        self.main_tab.AddPage(self.models, _("Models"))
        sizer_1.Add(self.main_tab, 2, wx.EXPAND, 0)
        sizer_1.Add(self.log, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()
        # end wxGlade

    def onOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory,
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )

        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            print("You chose the following file(s):")
            for path in paths:
                print(path)
        dlg.Destroy()

    def onDir(self, event):
        """
        Show the DirDialog and print the user's choice to stdout
        """
        dlg = wx.DirDialog(self, "Choose a directory:",
                           defaultPath=os.path.join(self.currentDirectory, "dataset"),
                           style=wx.DD_DEFAULT_STYLE 
                           #| wx.DD_DIR_MUST_EXIST
                           #| wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            print("You chose %s" % path)
        dlg.Destroy()
        return path

    def data_load_button_clicked(self, event):
        path = self.onDir(event)
        self.buildTree(self.data_tree, path)
   
    def data_tree_OnActivated(self, event):
        item = self.data_tree.GetFocusedItem()
 #     self.getStatistics(self.data_tree.GetItemData(item))
 #     def getStatistics(self, rootdirPath):
        dirPath = self.data_tree.GetItemData(item)
        self.text_ctrl_9.SetValue("")
        self.text_ctrl_9.write(dirPath)
        self.text_ctrl_10.SetValue("")
        self.text_ctrl_10.write("Not Implemented")
        self.text_ctrl_11.SetValue("")
        self.text_ctrl_11.write("Not Implemented")
        self.text_ctrl_12.SetValue("")
        self.text_ctrl_12.write("Not Implemented")
        self.text_ctrl_13.SetValue("")
        self.text_ctrl_13.write("Not Implemented")
        self.text_ctrl_15.SetValue("")
        self.text_ctrl_15.write("10")
        self.text_ctrl_16.SetValue("")
        self.text_ctrl_16.write("0")
        #datadirPath = os.path.join(rootdirPath, 'Data/')
        #file_list = os.listdir(datadirPath)


    def data_select_button_clicked(self, event):
        self.datasetID = datasetID = self.data_tree.GetFocusedItem()
        datasetName = self.data_tree.GetItemText(datasetID)
        parentID = self.data_tree.GetItemParent(datasetID)
        if parentID == self.data_tree.GetRootItem():
            print("Dataset '%s' is Selected!"%datasetName)

        # access to selected dataset path with this code
        datasetPath = self.data_tree.GetItemData(self.datasetID)

    def model_make_button_clicked(self, event):
        dlg = MyDialog(self, wx.ID_ANY, "")
        dlg.Show()


    def buildTree(self, tree, rootdirPath):
        def itemExist(tree, data, rootID):
            item, cookie = tree.GetFirstChild(rootID)
            while item.IsOk():
                if tree.GetItemData(item) == data:
                    return True
                item, cookie = tree.GetNextChild(rootID, cookie)
            return False

        if tree.IsEmpty() or not itemExist(tree, rootdirPath, tree.GetRootItem()):
            rootID = tree.AppendItem(tree.GetRootItem(), (os.path.basename(rootdirPath)))
            tree.SetItemData(rootID, rootdirPath)
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
                    
# end of class MyFrame
