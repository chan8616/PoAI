import wx
import wx.lib.inspection
import wx.lib.agw.multidirdialog as MDD

import os

import gettext
_ = gettext.gettext

class DataSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(DataSpecPage, self).__init__(parent, id)

        self.text_ctrl_2 = wx.TextCtrl(self, wx.ID_ANY, "2", style=wx.TE_READONLY)
        self.text_ctrl_3 = wx.TextCtrl(self, wx.ID_ANY, "3", style=wx.TE_READONLY)
        self.text_ctrl_4 = wx.TextCtrl(self, wx.ID_ANY, "4", style=wx.TE_READONLY)
        self.text_ctrl_5 = wx.TextCtrl(self, wx.ID_ANY, "5", style=wx.TE_READONLY)
        self.text_ctrl_6 = wx.TextCtrl(self, wx.ID_ANY, "6", style=wx.TE_READONLY)
#        self.combo_box_1 = wx.ComboBox(self, wx.ID_ANY, choices=["combo_box_1"], style=wx.CB_DROPDOWN)
#        self.text_ctrl_8 = wx.TextCtrl(self, wx.ID_ANY, "8", style=wx.TE_READONLY)
        #self.text_ctrl_7 = wx.TextCtrl(self, wx.ID_ANY, "7", style=wx.TE_READONLY)

        self.__do_layout()

    def setDataSpec(self, data_spec):
        self.text_ctrl_2.SetValue("")
        self.text_ctrl_2.write(data_spec['path'])
        self.text_ctrl_3.SetValue("")
        self.text_ctrl_3.write(str(data_spec['input_types']))
        self.text_ctrl_4.SetValue("")
        self.text_ctrl_4.write(str(data_spec['input_shapes']))
        self.text_ctrl_5.SetValue("")
        self.text_ctrl_5.write(data_spec['output_size'])
        self.text_ctrl_6.SetValue("")
        self.text_ctrl_6.write(str(len(data_spec['data']['train']['x'])) + ', ' + str(len(data_spec['data']['test']['x'])))
#        self.text_ctrl_8.SetValue("")
#        self.text_ctrl_8.write(str(len(data_spec['data']['test']['x']) / len(data_spec['data']['train']['x'])))

#        self.text_ctrl_2.SetValue("")
#        self.text_ctrl_2.write(data_spec['path'])

#    def __init__(self, parent, id, dict):
#        super(DataSpecPage, self).__init__(parent, id)
#
#        self.text_ctrl_2 = wx.TextCtrl(self, wx.ID_ANY, dict['path'] if 'path' in dict else "2", style=wx.TE_READONLY)
#        self.text_ctrl_3 = wx.TextCtrl(self, wx.ID_ANY, "3", style=wx.TE_READONLY)
#        self.text_ctrl_4 = wx.TextCtrl(self, wx.ID_ANY, "4", style=wx.TE_READONLY)
#        self.text_ctrl_5 = wx.TextCtrl(self, wx.ID_ANY, "5", style=wx.TE_READONLY)
#        self.combo_box_1 = wx.ComboBox(self, wx.ID_ANY, choices=["combo_box_1"], style=wx.CB_DROPDOWN)
#        self.text_ctrl_8 = wx.TextCtrl(self, wx.ID_ANY, "8")
#        self.text_ctrl_7 = wx.TextCtrl(self, wx.ID_ANY, "7")
#
#        self.__do_layout()
#
    def __do_layout(self):
        grid_sizer_1 = wx.GridBagSizer(5, 5)

        label_1 = wx.StaticText(self, wx.ID_ANY, _("Dataset Path"), style=wx.ALIGN_LEFT)
        label_1.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_1, (1, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_2, (1, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_2 = wx.StaticText(self, wx.ID_ANY, _("Input Type"), style=wx.ALIGN_LEFT)
        label_2.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_2, (2, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_3, (2, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_3 = wx.StaticText(self, wx.ID_ANY, _("Input Size"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_3.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_3, (3, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_4, (3, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_4 = wx.StaticText(self, wx.ID_ANY, _("Output Size"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_4.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_4, (4, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_5, (4, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_5 = wx.StaticText(self, wx.ID_ANY, _("Number of data (Train, Test)"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_5.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_5, (5, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        #grid_sizer_1.Add(self.combo_box_1, (5, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        grid_sizer_1.Add(self.text_ctrl_6, (5, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

#        label_6 = wx.StaticText(self, wx.ID_ANY, _("% of Testing"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
#        label_6.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
#        grid_sizer_1.Add(label_6, (6, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
#        grid_sizer_1.Add(self.text_ctrl_8, (6, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        # label_8 = wx.StaticText(self, wx.ID_ANY, _("Maximum Samples per Class"), style=wx.ALIGN_LEFT)
        # label_8.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        # grid_sizer_1.Add(label_8, (7, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        # grid_sizer_1.Add(self.text_ctrl_7, (7, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        self.SetSizer(grid_sizer_1)
#        grid_sizer_1.AddGrowableRow(18)
        #grid_sizer_1.AddGrowableCol(8)
        #grid_sizer_1.AddGrowableCol(22)

class ModelSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(ModelSpecPage, self).__init__(parent, id)

        self.text_ctrl_9 = wx.TextCtrl(self, wx.ID_ANY, "9", style=wx.TE_READONLY)
        self.text_ctrl_10 = wx.TextCtrl(self, wx.ID_ANY, "10", style=wx.TE_READONLY)
#        self.text_ctrl_11 = wx.TextCtrl(self, wx.ID_ANY, "11", style=wx.TE_READONLY)
#        self.text_ctrl_13 = wx.TextCtrl(self, wx.ID_ANY, "13")
#        self.text_ctrl_12 = wx.TextCtrl(self, wx.ID_ANY, "12", style=wx.TE_READONLY)

        self.__do_layout()

    def setModelSpec(self, model_spec):
        self.text_ctrl_9.SetValue("")
        self.text_ctrl_9.write(model_spec['network_type'])
        self.text_ctrl_10.SetValue("")
        self.text_ctrl_10.write(model_spec['input_type'])
#        self.text_ctrl_11.SetValue("")
#        self.text_ctrl_11.write(model_spec['input_size'])
#        self.text_ctrl_12.SetValue("")
#        self.text_ctrl_12.write(model_spec['output_size'])
#        self.text_ctrl_13.SetValue("")
#        self.text_ctrl_13.write(model_spec['type'])

#    def __init__(self, parent, id, dict):
#        super(ModelSpecPage, self).__init__(parent, id)
#
#        self.text_ctrl_9 = wx.TextCtrl(self, wx.ID_ANY, dict['type'] if 'type' in dict else "9")
#        self.text_ctrl_10 = wx.TextCtrl(self, wx.ID_ANY, dict['n_layer'] if 'n_layer' in dict else "10")
#        self.text_ctrl_11 = wx.TextCtrl(self, wx.ID_ANY, dict['input_size'] if 'image_size' in dict else "11")
#        self.text_ctrl_13 = wx.TextCtrl(self, wx.ID_ANY, "13")
#        self.text_ctrl_12 = wx.TextCtrl(self, wx.ID_ANY, dict['output_size'] if 'output_size' in dict else "12")
#
#        self.__do_layout()

    def __do_layout(self):
        grid_sizer_2 = wx.GridBagSizer(5, 5)

        label_9 = wx.StaticText(self, wx.ID_ANY, _("Network Type"), style=wx.ALIGN_LEFT)
        label_9.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_9, (1, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_9, (1, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_10 = wx.StaticText(self, wx.ID_ANY, _("Input Type"), style=wx.ALIGN_LEFT)
        label_10.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_10, (2, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_10, (2, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

#        label_11 = wx.StaticText(self, wx.ID_ANY, _("Input Size"), style=wx.ALIGN_LEFT)
#        label_11.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
#        grid_sizer_2.Add(label_11, (3, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
#        grid_sizer_2.Add(self.text_ctrl_11, (3, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

#        label_13 = wx.StaticText(self, wx.ID_ANY, _("X"), style=wx.ALIGN_CENTER)
#        grid_sizer_2.Add(label_13, (3, 21), (1, 1), wx.ALIGN_CENTER, 0)
#        grid_sizer_2.Add(self.text_ctrl_13, (3, 22), (1, 13), wx.EXPAND | wx.RIGHT, 20)

#        label_12 = wx.StaticText(self, wx.ID_ANY, _("Output Size"))
#        label_12.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
#        grid_sizer_2.Add(label_12, (4, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
#        grid_sizer_2.Add(self.text_ctrl_12, (4, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        self.SetSizer(grid_sizer_2)
#        grid_sizer_2.AddGrowableRow(18)
#        grid_sizer_2.AddGrowableCol(8)
#        grid_sizer_2.AddGrowableCol(22)

class TrainSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(TrainSpecPage, self).__init__(parent, id)

        self.combo_box_2 = wx.ComboBox(self, wx.ID_ANY, choices=["2"], style = wx.CB_DROPDOWN | wx.CB_READONLY)
        self.combo_box_3 = wx.ComboBox(self, wx.ID_ANY, choices=["3"], style = wx.CB_DROPDOWN | wx.CB_READONLY)
        self.combo_box_4 = wx.ComboBox(self, wx.ID_ANY, choices=["4"], style = wx.CB_DROPDOWN | wx.CB_READONLY)
        self.combo_box_6 = wx.ComboBox(self, wx.ID_ANY, choices=["6"], style = wx.CB_DROPDOWN)
        self.combo_box_7 = wx.ComboBox(self, wx.ID_ANY, choices=["7"], style = wx.CB_DROPDOWN)
        #self.combo_box_6 = wx.ComboBox(self, wx.ID_ANY, choices=["6"], style = wx.CB_DROPDOWN | wx.CB_READONLY)
        #self.text_ctrl_15 = wx.TextCtrl(self, wx.ID_ANY, "15")
        self.text_ctrl_16 = wx.TextCtrl(self, wx.ID_ANY, "26")

        self.text_ctrl_18 = wx.TextCtrl(self, wx.ID_ANY, "18")
        self.text_ctrl_19 = wx.TextCtrl(self, wx.ID_ANY, "19")
        #self.text_ctrl_20 = wx.TextCtrl(self, wx.ID_ANY, "20")
        self.text_ctrl_21 = wx.TextCtrl(self, wx.ID_ANY, "21")
        self.text_ctrl_22 = wx.TextCtrl(self, wx.ID_ANY, "22")

        self.__do_layout()
        self.__do_binds()

    def setTrainSpec(self, train_spec):
        self.train_spec = train_spec

        self.combo_box_2.Delete(0)
        #for i, model_name in enumerate(train_spec['model_names']):
        for i, model_name in enumerate(train_spec['model_dict'].keys()):
            self.combo_box_2.Insert(model_name, 0)
        if 'trained' in train_spec:
            idx = self.combo_box_2.FindString(train_spec['model_name'])
            self.combo_box_2.SetSelection(idx)
            self.SetCheckpoint(train_spec['model_name'])

        self.combo_box_3.Delete(0)
        #for i, dataset_name in enumerate(train_spec['dataset_names']):
        for i, dataset_name in enumerate(train_spec['dataset_dict'].keys()):
            self.combo_box_3.Insert(dataset_name, 0)
        if 'trained' in train_spec:
            idx = self.combo_box_3.FindString(train_spec['trained']['dataset'])
            self.combo_box_3.SetSelection(idx)

        from utils.util import gpu_inspection

        self.combo_box_4.Delete(0)
        num_gpus = gpu_inspection()
        for gpu in range(num_gpus): # -1 means CPU
            self.combo_box_4.Insert(str(gpu),0)
        self.combo_box_4.Insert('cpu', 0)
        self.combo_box_4.SetSelection(0)

        self.combo_box_6.Delete(0)
        self.combo_box_6.Insert("New",0)
        if 'trained' in train_spec:
            self.train_spec['checkpoint_name'] = train_spec['trained']['name']
            self.setTrainSpec_checkpoint_name()

        self.combo_box_7.Delete(0)
        for name in train_spec['solver_list']:
            self.combo_box_7.Insert(name, 0)
        if 'trained' in train_spec:
            idx = self.combo_box_7.FindString(train_spec['trained']['optimizer'])
            self.combo_box_7.SetSelection(idx)
        else:
            self.combo_box_7.SetSelection(0)

        #self.text_ctrl_15.SetValue("")
        #self.text_ctrl_15.write(train_spec['checkpoint_name'])
        self.text_ctrl_16.SetValue("")
        self.text_ctrl_16.write(train_spec['max_epochs'] if 'trained' not in train_spec else str(train_spec['trained']['epochs']))
        self.text_ctrl_18.SetValue("")
        self.text_ctrl_18.write(train_spec['batch_size'] if 'trained' not in train_spec else str(train_spec['trained']['batch_size']))
        #self.text_ctrl_20.SetValue("")
        #self.text_ctrl_20.write(train_spec['optimizer'])
        self.text_ctrl_21.SetValue("")
        self.text_ctrl_21.write(train_spec['learning_rate'] if 'trained' not in train_spec else str(train_spec['trained']['learning_rate']))
        self.text_ctrl_19.SetValue("")
        self.text_ctrl_19.write(train_spec['interval'])# if 'trained' not in train_spec else str(train_spec['trained']['learning_rate']))
        self.text_ctrl_22.SetValue("")
        self.text_ctrl_22.write(train_spec['seed'])
#
#    def __init__(self, parent, id, dict):
#        super(TrainSpecPage, self).__init__(parent, id)
#
#        #self.text_ctrl_14 = wx.TextCtrl(self, wx.ID_ANY, "14")
#        self.combo_box_2 = wx.ComboBox(self, wx.ID_ANY, choices=dict['model_names'] if 'model_names' in dict else ["2"], style = wx.CB_DROPDOWN | wx.CB_READONLY)
#        self.combo_box_3 = wx.ComboBox(self, wx.ID_ANY, choices=dict['dataset_names'] if 'dataset_names' in dict else ["3"], style = wx.CB_DROPDOWN | wx.CB_READONLY)
#        self.text_ctrl_15 = wx.TextCtrl(self, wx.ID_ANY, "15")
#        self.text_ctrl_16 = wx.TextCtrl(self, wx.ID_ANY, str(dict['max_iter']) if 'max_iter' in dict else "26")
#        self.text_ctrl_18 = wx.TextCtrl(self, wx.ID_ANY, str(dict['batch_size']) if 'batch_size' in dict else "18")
#        self.text_ctrl_20 = wx.TextCtrl(self, wx.ID_ANY, dict['optimizer'] if 'optimizer' in dict else "20")
#        self.text_ctrl_21 = wx.TextCtrl(self, wx.ID_ANY, str(dict['lr']) if 'lr' in dict else "21")
#        self.text_ctrl_19 = wx.TextCtrl(self, wx.ID_ANY, str(dict['validation interval']) if 'validation interval' in dict else "19")
#        self.text_ctrl_22 = wx.TextCtrl(self, wx.ID_ANY, str(dict['seed']) if 'seed' in dict else "22")
#        self.text_ctrl_23 = wx.TextCtrl(self, wx.ID_ANY, "23")
#
#        self.__do_layout()

    def __do_layout(self):
        grid_sizer_3 = wx.GridBagSizer(0, 0)

        label_14 = wx.StaticText(self, wx.ID_ANY, _("Model select"))
        label_14.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_14, (1, 1), (1, 10), 0, 0)
        grid_sizer_3.Add(self.combo_box_2, (2, 1), (1, 18), wx.EXPAND, 0)

        label_19 = wx.StaticText(self, wx.ID_ANY, _("Sanpshot ratio (in epochs)"))
        label_19.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_19, (1, 21), (1, 10), 0, 0)
        #grid_sizer_3.Add(self.text_ctrl_14, (2, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_19, (2, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)

        label_15 = wx.StaticText(self, wx.ID_ANY, _("Checkpoint name (Select trained or Write new name)"))
        label_15.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_15, (4, 1), (1, 10), 0, 0)

        label_20 = wx.StaticText(self, wx.ID_ANY, _("Solver type"))
        label_20.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_20, (4, 21), (1, 10), 0, 0)
        #grid_sizer_3.Add(self.text_ctrl_15, (5, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.combo_box_6, (5, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.combo_box_7, (5, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)
        #grid_sizer_3.Add(self.text_ctrl_20, (5, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)

        label_16 = wx.StaticText(self, wx.ID_ANY, _("Training epochs"))
        label_16.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_16, (7, 1), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_16, (8, 1), (1, 18), wx.EXPAND, 0)

        label_21 = wx.StaticText(self, wx.ID_ANY, _("Base Learning rate"))
        label_21.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_21, (7, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_21, (8, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)

        label_17 = wx.StaticText(self, wx.ID_ANY, _("Dataset select"))
        label_17.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_17, (10, 1), (1, 10), 0, 0)

        label_22 = wx.StaticText(self, wx.ID_ANY, _("Random seed"))
        label_22.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_22, (10, 21), (1, 10), 0, 0)
        #grid_sizer_3.Add(self.text_ctrl_17, (11, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.combo_box_3, (11, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_22, (11, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)

        label_18 = wx.StaticText(self, wx.ID_ANY, _("Batch size"))
        label_18.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_18, (13, 1), (1, 10), 0, 0)

        label_23 = wx.StaticText(self, wx.ID_ANY, _("Select which GPU[s] you would like to use"))
        label_23.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_23, (13, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_18, (14, 1), (1, 18), wx.EXPAND, 0)
        #grid_sizer_3.Add(self.text_ctrl_23, (14, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)
        grid_sizer_3.Add(self.combo_box_4, (14, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)

        self.SetSizer(grid_sizer_3)

    def __do_binds(self):
        self.combo_box_2.Bind(wx.EVT_COMBOBOX, self.OnModelSelect)
        self.combo_box_3.Bind(wx.EVT_COMBOBOX, self.OnDatasetSelect)

    def ModelSelected(self):
        return self.combo_box_2.GetSelection()
    def DatasetSelected(self):
        return self.combo_box_3.GetSelection()

    def OnModelSelect(self, event):
        if self.DatasetSelected() != wx.NOT_FOUND:
            self.SetCheckpointname()

        model_name = event.GetString()
        self.SetCheckpoint(model_name)

        return
        for i in range(1, self.combo_box_6.GetCount()):
            self.combo_box_6.Delete(1)
        #print(event.GetSelection())
        for data_name in self.train_spec['trained_model_dict'][model_name].keys():
            for trained_model_name in self.train_spec['trained_model_dict'][model_name][data_name].keys():
                self.combo_box_6.Insert(data_name + '/' + trained_model_name, 1)
        #print(self.train_spec['trained_model_dict'].keys())

#        self.train_spec['checkpoint_name'] = event.GetString().split("_")[0] + "_" + self.train_spec['checkpoint_name'].split("_")[-1]
#        self.setTrainSpec_checkpoint_name()

    def OnDatasetSelect(self, event):
        if self.ModelSelected() != wx.NOT_FOUND:
            self.train_spec['checkpoint_name'] = event.GetString() + '_1'
            self.SetCheckpointname()

    def SetCheckpoint(self, model_name):
        for i in range(1, self.combo_box_6.GetCount()):
            self.combo_box_6.Delete(1)
        #print(event.GetSelection())
        for data_name in self.train_spec['trained_model_dict'][model_name].keys():
            for trained_model_name in self.train_spec['trained_model_dict'][model_name][data_name].keys():
                self.combo_box_6.Insert(trained_model_name, 1)
                #self.combo_box_6.Insert(data_name + '/' + trained_model_name, 1)

    def SetCheckpointname(self):
        #print('setcheckpointname', self.combo_box_6.GetSelection())
        if self.combo_box_6.GetSelection() == 0 or self.combo_box_6.GetSelection() == wx.NOT_FOUND: # New or another name
            #print("checkpoint name", self.train_spec['checkpoint_name'])
            while self.combo_box_6.FindString(self.train_spec['checkpoint_name']) is not wx.NOT_FOUND:
                #print("name founded")
                try:
                    num = int(self.train_spec['checkpoint_name'].split("_")[-1])
                    idx = self.train_spec['checkpoint_name'].rfind("_")
                    self.train_spec['checkpoint_name'] = self.train_spec['checkpoint_name'][:idx]
                    #print("num founded", num, idx)
                except ValueError:
                    num = 0
                    #print("num not founded")
                num += 1
                self.train_spec['checkpoint_name'] += "_" + str(num)
#                while self.combo_box_6.FindString(self.train_spec['checkpoint_name']) != wx.NOT_FOUND:
#                    print("name founded")
#                    self.train_spec['checkpoint_name'] += "_" + str(num)
#                    num += 1
            self.setTrainSpec_checkpoint_name()

    def OnCheckpointnameSelected(self, event):
        print("Checkpointname Selected")
        if event.GetSelection() == 0:
            self.combo_box_6.SetFont(wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        else:
            self.combo_box_6.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        self.setTrainSpec_checkpoint_name()

#    def setTrainSpec_(self, train_spec):
#        self.combo_box_2.Delete(0)
#        for i, model_name in enumerate(train_spec['model_names']):
#            self.combo_box_2.Insert(model_name, 0)
#        self.combo_box_3.Delete(0)
#        for i, dataset_name in enumerate(train_spec['dataset_names']):
#            self.combo_box_3.Insert(dataset_name, 0)
#        self.combo_box_4.Delete(0)
#        for i, gpu in enumerate(train_spec['gpus']):
#            self.combo_box_4.Insert(gpu, 0)
#        self.text_ctrl_15.SetValue("")
#        self.text_ctrl_15.write(train_spec['checkpoint_name'])
#        self.text_ctrl_16.SetValue("")
#        self.text_ctrl_16.write(train_spec['max_iter'])
#        self.text_ctrl_18.SetValue("")
#        self.text_ctrl_18.write(train_spec['batch_size'])
#        self.text_ctrl_20.SetValue("")
#        self.text_ctrl_20.write(train_spec['optimizer'])
#        self.text_ctrl_21.SetValue("")
#        self.text_ctrl_21.write(train_spec['learning_rate'])
#        self.text_ctrl_19.SetValue("")
#        self.text_ctrl_19.write(train_spec['interval'])
#        self.text_ctrl_22.SetValue("")
#        self.text_ctrl_22.write(train_spec['seed'])

    def setTrainSpec_checkpoint_name(self):
        self.combo_box_6.SetValue(self.train_spec['checkpoint_name'])
        #self.combo_box_6.SetString(0, self.train_spec['checkpoint_name'])
        #self.text_ctrl_15.write(self.train_spec['checkpoint_name'])
        pass
        #self.text_ctrl_15.SetValue("")
        #self.text_ctrl_15.write(self.train_spec['checkpoint_name'])

    def getTrainSpec(self):
        ### return dict
        ### model_name, dataset_name, gpu, trained_model_name, checkpoint_name,
        ### max_ecpochs, batch_size, optimizer, learning_rate, interval, speed
        spec = {}

        model_name_idx = self.combo_box_2.GetSelection()
        assert model_name_idx != wx.NOT_FOUND, "[!] select model"
        #self.train_spec['model_name'] = self.combo_box_2.GetStringSelection()
        spec['model_name'] = self.combo_box_2.GetStringSelection()

        dataset_name_idx = self.combo_box_3.GetSelection()
        assert model_name_idx != wx.NOT_FOUND, "[!] select dataset"
        #self.train_spec['dataset_name'] = self.combo_box_3.GetStringSelection()
        spec['dataset_name'] = self.combo_box_3.GetStringSelection()

        gpu_idx = self.combo_box_4.GetSelection()
        assert gpu_idx != wx.NOT_FOUND, "[!] select gpu"
        #self.train_spec['gpu'] = self.combo_box_4.GetStringSelection()
        spec['gpu'] = self.combo_box_4.GetStringSelection()

        train_model_name = self.combo_box_6.GetValue()
        assert train_model_name != "", "select trained_model or write checkpoint name(combo box 6)"
        #self.train_spec['trained_model_name'] = None \
        spec['trained_model_name'] = None \
            if self.combo_box_6.GetSelection() == wx.NOT_FOUND \
            else self.combo_box_6.GetStringSelection()

        train_model_name_idx = self.combo_box_7.GetSelection()
        assert train_model_name_idx != wx.NOT_FOUND, "select solver(combo box 7)"
        #self.train_spec['optimizer'] = self.combo_box_7.GetStringSelection()
        spec['optimizer'] = self.combo_box_7.GetStringSelection()

        spec['checkpoint_name'] = self.combo_box_6.GetValue()
        spec['max_epochs'] = self.text_ctrl_16.GetLineText(0)
        spec['batch_size'] = self.text_ctrl_18.GetLineText(0)
        #spec['optimizer'] = self.text_ctrl_20.GetLineText(0)
        spec['learning_rate'] = self.text_ctrl_21.GetLineText(0)
        spec['interval'] = self.text_ctrl_19.GetLineText(0)
        spec['seed'] = self.text_ctrl_22.GetLineText(0)
        return spec


#        self.train_spec['checkpoint_name'] = self.combo_box_6.GetValue()
#        self.train_spec['max_epochs'] = self.text_ctrl_16.GetLineText(0)
#        self.train_spec['batch_size'] = self.text_ctrl_18.GetLineText(0)
#        #self.train_spec['optimizer'] = self.text_ctrl_20.GetLineText(0)
#        self.train_spec['learning_rate'] = self.text_ctrl_21.GetLineText(0)
#        self.train_spec['interval'] = self.text_ctrl_19.GetLineText(0)
#        self.train_spec['seed'] = self.text_ctrl_22.GetLineText(0)
#        return self.train_spec

class TestSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(TestSpecPage, self).__init__(parent, id)

        self.combo_box_5 = wx.ComboBox(self, wx.ID_ANY, choices=["5"], style = wx.CB_DROPDOWN | wx.CB_READONLY) # model select
        self.button_1 = wx.Button(self, wx.ID_ANY, _("Browse")) # file browser
        self.button_2 = wx.Button(self, wx.ID_ANY, _("Browse")) # folder browser
        self.text_ctrl_24 = wx.TextCtrl(self, wx.ID_ANY, "24") # file names
        self.text_ctrl_25 = wx.TextCtrl(self, wx.ID_ANY, "25") # path
        self.text_ctrl_26 = wx.TextCtrl(self, wx.ID_ANY, "26") # number of images
        self.text_ctrl_27 = wx.TextCtrl(self, wx.ID_ANY, "27") # image sizes
        self.text_ctrl_28 = wx.TextCtrl(self, wx.ID_ANY, "28") # image types
        self.upload_list = None

        self.__do_layout()
        self.__do_binds()

    def __do_layout(self):
        grid_sizer_4 = wx.GridBagSizer(0, 0)

        label_14 = wx.StaticText(self, wx.ID_ANY, _("Model select"))
        label_14.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_14, (1, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.combo_box_5, (1, 11), (1, 21), wx.EXPAND | wx.RIGHT, 30)

        label_24 = wx.StaticText(self, wx.ID_ANY, _("Upload images"))
        label_24.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, "Ubuntu"))
        grid_sizer_4.Add(label_24, (3, 1), (1, 7), 0, 0)
        grid_sizer_4.Add(self.button_1, (3, 8), (1, 4), 0, 0)

        label_30 = wx.StaticText(self, wx.ID_ANY, _("Upload folders"))
        label_30.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, "Ubuntu"))
        grid_sizer_4.Add(label_30, (5, 1), (1, 7), 0, 0)
        grid_sizer_4.Add(self.button_2, (5, 8), (1, 4), 0, 0)

        label_29 = wx.StaticText(self, wx.ID_ANY, _("File names"))
        label_29.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_29, (7, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_24, (7, 11), (1, 21), wx.EXPAND | wx.RIGHT, 30)

        label_25 = wx.StaticText(self, wx.ID_ANY, _("File path"))
        label_25.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_25, (9, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_25, (9, 11), (1, 21), wx.EXPAND | wx.RIGHT, 30)

        #label_26 = wx.StaticText(self, wx.ID_ANY, _("Number of images use from the file"))
        label_26 = wx.StaticText(self, wx.ID_ANY, _("Number of data"))
        label_26.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_26, (11, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_26, (11, 11), (1, 21), wx.EXPAND | wx.RIGHT, 30)

        #label_27 = wx.StaticText(self, wx.ID_ANY, _("Leave blank to use all"))
        #label_27 = wx.StaticText(self, wx.ID_ANY, _("Leave blank to use all"))
        #label_27.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.LIGHT, 0, ""))
        #grid_sizer_4.Add(label_27, (14, 1), (1, 10), 0, 0)

        #label_28 = wx.StaticText(self, wx.ID_ANY, _("Number of images to show per category"))
        label_28 = wx.StaticText(self, wx.ID_ANY, _("data sizes"))
        label_28.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_28, (15, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_27, (15, 11), (1, 21), wx.EXPAND | wx.RIGHT, 30)

        label_27 = wx.StaticText(self, wx.ID_ANY, _("data type"))
        label_27.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_27, (13, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.text_ctrl_28, (13, 11), (1, 21), wx.EXPAND | wx.RIGHT, 30)

        grid_sizer_4.AddGrowableCol(20)
        self.SetSizer(grid_sizer_4)

    def __do_binds(self):
        self.button_1.Bind(wx.EVT_BUTTON, self.OnFileDialog)
        self.button_2.Bind(wx.EVT_BUTTON, self.OnDirDialog)

    def OnFileDialog(self, event):
        dlg = wx.FileDialog(self, "", defaultDir=self.test_spec['default_dataset_path'],
                wildcard="(*.*)|*.*",
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

        if dlg.ShowModal() == wx.ID_OK:
            self.upload_list = dlg.GetPaths()
            self.text_ctrl_24.SetValue(str(dlg.GetFilenames()))
            self.text_ctrl_25.SetValue("")

    def OnDirDialog(self, event):
        #dlg = MDD.MultiDirDialog(None, title="", defaultPath="",#self.test_spec['default_dataset_path'],
        #    agwStyle=MDD.DD_MULTIPLE | MDD.DD_DIR_MUST_EXIST)
        dlg = wx.DirDialog(self, "", defaultPath=self.test_spec['default_dataset_path'],
            style=wx.DD_DIR_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            #self.upload_list = dlg.GetPaths()
            self.upload_list = [dlg.GetPath()]
            self.text_ctrl_24.SetValue("")
            self.text_ctrl_25.SetValue(str(dlg.GetPath()))

    def setTestSpec(self, test_spec):
        self.test_spec = test_spec

        self.text_ctrl_24.SetValue("")
        self.text_ctrl_25.SetValue("")
        self.text_ctrl_26.SetValue("")
        self.text_ctrl_27.SetValue("")
        self.text_ctrl_28.SetValue("")

        self.combo_box_5.Delete(0)
        for model_name in test_spec['trained_model_dict'].keys():
            for data_name in test_spec['trained_model_dict'][model_name].keys():
                for trained_model_name in test_spec['trained_model_dict'][model_name][data_name].keys():
                    self.combo_box_5.Insert(model_name + '/' + data_name + '/' + trained_model_name, 0)
        #self.text_ctrl_24.SetValue("")
        #self.text_ctrl_25.SetValue("")

#    def setTestSpec_dataset(self):
#        self.test_spec['dataset_path']
#        self.test_spec['dataset_name']

    def getTestSpec(self):
        ### return dict
        ### model_name, upload_list
        spec = {}
        idx = self.combo_box_5.GetSelection()
        assert idx != wx.NOT_FOUND, "[!] Not Found"
        trained_model_name = self.combo_box_5.GetStringSelection()
        spec['model_name'], spec['dataset_name'], spec['trained_model_name'] = trained_model_name.split('/')
        spec['upload_list'] = self.upload_list
        #self.test_spec += [self.text_ctrl_26.GetLineText(0)]
        #self.test_spec += [self.text_ctrl_27.GetLineText(0)]

        return spec

class MyNotebook(wx.lib.agw.aui.auibook.AuiNotebook):
    def __init__(self, *args, **kwds):
		# Auinotebook
        super(MyNotebook, self).__init__(*args, **kwds)

        self.data_spec_count = 0
        self.model_spec_count = 0
        self.train_spec_count = 0
        self.test_spec_count = 0
        self.__do_layout()
        # Auinotebook end

    def __do_layout(self):
         #self.notebook_1.AddPage(self.DataSpec, _("Dataset Spec"))
#        for i in range(len(self.panel_1_list)):
#            self.notebook_1.AddPage(self.panel_1_list[i], _("Dataset Spec"))
        #self.notebook_1.AddPage(self.ModelSpec, _("Model Spec"))
        #self.notebook_1.AddPage(self.Makeamodel, _("Make a model"))
#        self.AddPage(self.Testmodelsindleimage, _("Test model(sindle image)"))
#        self.AddPage(self.Testmodelimagefolder, _("Test model(image folder)"))
        self.Layout()

   #def createTestSpecPanel(self, parent, id, dict):
    def createTestSpecPanel(self, parent, id):
        self.test_spec_count += 1
        test_spec_panel = TestSpecPage(parent, id)
        self.AddPage(test_spec_panel, _("Test Spec %d"%self.test_spec_count), select=True)
        return test_spec_panel

    #def createTrainSpecPanel(self, parent, id, dict):
    def createTrainSpecPanel(self, parent, id):
        self.train_spec_count += 1
        train_spec_panel = TrainSpecPage(parent, id)
        self.AddPage(train_spec_panel, _("Train Spec %d"%self.train_spec_count), select=True)
        return train_spec_panel

    #def createModelSpecPanel(self, parent, id, dict):
    def createModelSpecPanel(self, parent, id):
        self.model_spec_count += 1
        model_spec_panel = ModelSpecPage(parent, id)
        self.AddPage(model_spec_panel, _("Model Spec %d"%self.model_spec_count), select=True)
        return model_spec_panel

    #def createDataSpecPanel(self, parent, id, dict):
    def createDataSpecPanel(self, parent, id):
        self.data_spec_count += 1
        data_spec_panel = DataSpecPage(parent, id)
        self.AddPage(data_spec_panel, _("Data Spec %d"%self.data_spec_count), select=True)
        return data_spec_panel

    def getRunSpec(self):
        ### return page, phase, args
        ###             (train): max_epochs, learning_rate, seed, batch_size, interval,
        ###                     learning_rate, dataset_list, model_list, dataset_names, model_names,
        ###                     trained_model_dict, trained_model_names_dict, checkpoint_name, solver_dict,
        ###
        ###                     model_name, dataset_name, gpu, trained_model_name, checkpoint_name,
        ###                     max_ecpochs, batch_size, optimizer, learning_rate, interval, speed
        ###
        ###             (test): model_list, model_names, trained_model_list_names,
        ###
        ###                     model_name, upload_list

        page = self.GetPage(self.GetSelection())
        if isinstance(page, TrainSpecPage):
            print("Train Start")
            phase = 'Train'
            spec = page.getTrainSpec()
            ### return dict
            ### model_name, dataset_name, gpu, trained_model_name, checkpoint_name,
            ### max_ecpochs, batch_size, optimizer, learning_rate, interval, speed
        elif isinstance(page, TestSpecPage):
            print("Test Start")
            phase = 'Test'
            spec = page.getTestSpec()
            ### return dict
            ### model_name, upload_list
        else:
            return None

        return page, phase, spec

#         if isinstance(page, TrainSpecPage):
#             spec = ['train']
#             # model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
#             idx = page.combo_box_2.GetSelection() # model_name
#             if idx == wx.NOT_FOUND:
#                 print("select combo box 2")
#             else:
#                 spec += [page.combo_box_2.GetStringSelection()]
#             idx = page.combo_box_3.GetSelection() # dataset name
#             if idx == wx.NOT_FOUND:
#                 print("select combo box 3")
#             else:
#                 spec += [page.combo_box_3.GetStringSelection()]
#             idx = page.combo_box_4.GetSelection() # GPU_selection
#             if idx == wx.NOT_FOUND:
#                 print("select combo box 4")
#             else:
#                 spec += [page.combo_box_4.GetStringSelection()]
#             spec += [page.text_ctrl_15.GetLineText(0)] # checkpoint_name
#             spec += [page.text_ctrl_16.GetLineText(0)] # max_iter
#             spec += [page.text_ctrl_18.GetLineText(0)] # batch_size
#             spec += [page.text_ctrl_20.GetLineText(0)] # optimizer
#             spec += [page.text_ctrl_21.GetLineText(0)] # learning_rate
#             spec += [page.text_ctrl_19.GetLineText(0)] # interval
#             spec += [page.text_ctrl_22.GetLineText(0)] # random_seed
#             # spec += [page.text_ctrl_23.GetLineText(0)] # ?
#
#             return spec
#         elif isinstance(page, TestSpecPage):
#             spec = ['test']
#             idx = page.combo_box_5.GetSelection()
#             if idx == wx.NOT_FOUND:
#                 print("select combo box 5")
#             else:
#                 spec += [page.combo_box_5.GetStringSelection()]
#
# #            spec['model'] =
# #            spec['data'] =
# #
#             return spec
#         else:
#             return False

    def isOnTrainSpec(self):
        page = self.GetPage(self.GetSelection())
        return isinstance(page, TrainSpecPage)
    def isOnTestSpec(self):
        page = self.GetPage(self.GetSelection())
        return isinstance(page, TestSpecPage)

#    def OnTabClicked(self, event):
#        super(MyNotebook, self).OnTabClicked(event)
        #print('tab clicked', event, event.GetNotifyEvent())
        #event.Skip()
