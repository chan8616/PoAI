import wx
import wx.lib.inspection

class DataSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(DataSpecPage, self).__init__(parent, id)

        self.text_ctrl_2 = wx.TextCtrl(self, wx.ID_ANY, "2", style=wx.TE_READONLY)
        self.text_ctrl_3 = wx.TextCtrl(self, wx.ID_ANY, "3", style=wx.TE_READONLY)
        self.text_ctrl_4 = wx.TextCtrl(self, wx.ID_ANY, "4", style=wx.TE_READONLY)
        self.text_ctrl_5 = wx.TextCtrl(self, wx.ID_ANY, "5", style=wx.TE_READONLY)
#        self.text_ctrl_6 = wx.TextCtrl(self, wx.ID_ANY, "6", style=wx.TE_READONLY)
        self.combo_box_1 = wx.ComboBox(self, wx.ID_ANY, choices=["combo_box_1"], style=wx.CB_DROPDOWN)
        self.text_ctrl_8 = wx.TextCtrl(self, wx.ID_ANY, "8")
        self.text_ctrl_7 = wx.TextCtrl(self, wx.ID_ANY, "7")

        self.__do_layout()

    def setDataSpec(self, data_spec):
        self.text_ctrl_2.SetValue("")
        self.text_ctrl_2.write(data_spec['path'])

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

        label_2 = wx.StaticText(self, wx.ID_ANY, _("Number of Classes (Output size)"), style=wx.ALIGN_LEFT)
        label_2.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_2, (2, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_3, (2, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_3 = wx.StaticText(self, wx.ID_ANY, _("Number of Images"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_3.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_3, (3, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_4, (3, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_4 = wx.StaticText(self, wx.ID_ANY, _("Image Size (Input shape)"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_4.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_4, (4, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_5, (4, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_5 = wx.StaticText(self, wx.ID_ANY, _("Image Type"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_5.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_5, (5, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.combo_box_1, (5, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_6 = wx.StaticText(self, wx.ID_ANY, _("% of Testing"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_6.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_6, (6, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_8, (6, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_8 = wx.StaticText(self, wx.ID_ANY, _("Maximum Samples per Class"), style=wx.ALIGN_LEFT)
        label_8.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_8, (7, 1), (1, 10), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_7, (7, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        self.SetSizer(grid_sizer_1)
#        grid_sizer_1.AddGrowableRow(18)
        #grid_sizer_1.AddGrowableCol(8)
        #grid_sizer_1.AddGrowableCol(22)

class ModelSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(ModelSpecPage, self).__init__(parent, id)

        self.text_ctrl_9 = wx.TextCtrl(self, wx.ID_ANY, "9")
        self.text_ctrl_10 = wx.TextCtrl(self, wx.ID_ANY, "10")
        self.text_ctrl_11 = wx.TextCtrl(self, wx.ID_ANY, "11")
#        self.text_ctrl_13 = wx.TextCtrl(self, wx.ID_ANY, "13")
        self.text_ctrl_12 = wx.TextCtrl(self, wx.ID_ANY, "12")

        self.__do_layout()
    
    def setModelSpec(self, model_spec):
        self.text_ctrl_9.SetValue("")
        self.text_ctrl_9.write(model_spec['type'])
        self.text_ctrl_10.SetValue("")
        self.text_ctrl_10.write(model_spec['n_layer'])
        self.text_ctrl_11.SetValue("")
        self.text_ctrl_11.write(model_spec['input_size'])
        self.text_ctrl_12.SetValue("")
        self.text_ctrl_12.write(model_spec['output_size'])
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

        label_10 = wx.StaticText(self, wx.ID_ANY, _("Number of Layers"), style=wx.ALIGN_LEFT)
        label_10.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_10, (2, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_10, (2, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

        label_11 = wx.StaticText(self, wx.ID_ANY, _("Image Size (Input)"), style=wx.ALIGN_LEFT)
        label_11.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_11, (3, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_11, (3, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

#        label_13 = wx.StaticText(self, wx.ID_ANY, _("X"), style=wx.ALIGN_CENTER)
#        grid_sizer_2.Add(label_13, (3, 21), (1, 1), wx.ALIGN_CENTER, 0)
#        grid_sizer_2.Add(self.text_ctrl_13, (3, 22), (1, 13), wx.EXPAND | wx.RIGHT, 20)

        label_12 = wx.StaticText(self, wx.ID_ANY, _("Number of Classes / Demension (Output)"))
        label_12.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_12, (4, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_12, (4, 11), (1, 27), wx.EXPAND | wx.RIGHT, 20)

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
        self.text_ctrl_15 = wx.TextCtrl(self, wx.ID_ANY, "15")
        self.text_ctrl_16 = wx.TextCtrl(self, wx.ID_ANY, "26")

        self.text_ctrl_18 = wx.TextCtrl(self, wx.ID_ANY, "18")
        self.text_ctrl_19 = wx.TextCtrl(self, wx.ID_ANY, "19")
        self.text_ctrl_20 = wx.TextCtrl(self, wx.ID_ANY, "20")
        self.text_ctrl_21 = wx.TextCtrl(self, wx.ID_ANY, "21")
        self.text_ctrl_22 = wx.TextCtrl(self, wx.ID_ANY, "22")

        self.__do_layout()
        self.__do_binds()

    def __do_layout(self):
        grid_sizer_3 = wx.GridBagSizer(0, 0)

        label_14 = wx.StaticText(self, wx.ID_ANY, _("Model select"))
        label_14.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_14, (1, 1), (1, 10), 0, 0)
        grid_sizer_3.Add(self.combo_box_2, (2, 1), (1, 18), wx.EXPAND, 0)

        label_19 = wx.StaticText(self, wx.ID_ANY, _("Sanpshot interval(in epochs)"))
        label_19.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_19, (1, 21), (1, 10), 0, 0)
        #grid_sizer_3.Add(self.text_ctrl_14, (2, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_19, (2, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30) 

        label_15 = wx.StaticText(self, wx.ID_ANY, _("Checkpoint name"))
        label_15.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_15, (4, 1), (1, 10), 0, 0)

        label_20 = wx.StaticText(self, wx.ID_ANY, _("Solver type"))
        label_20.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_20, (4, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_15, (5, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_20, (5, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30) 

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
        self.combo_box_2.Bind(wx.EVT_COMBOBOX, self.OnModelSelected)
        self.combo_box_3.Bind(wx.EVT_COMBOBOX, self.OnDatasetSelected)

    def OnModelSelected(self, event):
        self.train_spec['checkpoint_name'] = event.GetString().split("_")[0] + "_" + self.train_spec['checkpoint_name'].split("_")[-1]
        self.setTrainSpec_checkpoint_name()

    def OnDatasetSelected(self, event):
        self.train_spec['checkpoint_name'] = self.train_spec['checkpoint_name'].split("_")[0] + "_" + event.GetString().split("_")[0]
        self.setTrainSpec_checkpoint_name()
        
    def setTrainSpec(self, train_spec):
        self.train_spec = train_spec

        self.combo_box_2.Delete(0)
        for i, model_name in enumerate(train_spec['model_names']):
            self.combo_box_2.Insert(model_name, 0)
        self.combo_box_3.Delete(0)
        for i, dataset_name in enumerate(train_spec['dataset_names']):
            self.combo_box_3.Insert(dataset_name, 0)
        self.combo_box_4.Delete(0)
        for i, gpu in enumerate(train_spec['gpus']):
            self.combo_box_4.Insert(gpu, 0)
        self.text_ctrl_15.SetValue("")
        self.text_ctrl_15.write(train_spec['checkpoint_name'])
        self.text_ctrl_16.SetValue("")
        self.text_ctrl_16.write(train_spec['max_iter'])
        self.text_ctrl_18.SetValue("")
        self.text_ctrl_18.write(train_spec['batch_size'])
        self.text_ctrl_20.SetValue("")
        self.text_ctrl_20.write(train_spec['optimizer'])
        self.text_ctrl_21.SetValue("")
        self.text_ctrl_21.write(train_spec['learning_rate'])
        self.text_ctrl_19.SetValue("")
        self.text_ctrl_19.write(train_spec['interval'])
        self.text_ctrl_22.SetValue("")
        self.text_ctrl_22.write(train_spec['seed'])

    def setTrainSpec_checkpoint_name(self):
        self.text_ctrl_15.SetValue("")
        self.text_ctrl_15.write(self.train_spec['checkpoint_name'])
    
    def getTrainSpec(self):
        spec = []
        idx = self.combo_box_2.GetSelection()
        if idx == wx.NOT_FOUND:
            print("select combo box 2")
        else:
            spec += [self.combo_box_2.GetStringSelection()]
        idx = self.combo_box_3.GetSelection()
        if idx == wx.NOT_FOUND:
            print("select combo box 3")
        else:
            spec += [self.combo_box_3.GetStringSelection()]
        idx = self.combo_box_4.GetSelection()
        if idx == wx.NOT_FOUND:
            print("select combo box 4")
        else:
            spec += [self.combo_box_4.GetStringSelection()]
        spec += [self.text_ctrl_15.GetLineText(0)]
        spec += [self.text_ctrl_16.GetLineText(0)]
        spec += [self.text_ctrl_18.GetLineText(0)]
        spec += [self.text_ctrl_20.GetLineText(0)]
        spec += [self.text_ctrl_21.GetLineText(0)]
        spec += [self.text_ctrl_19.GetLineText(0)]
        spec += [self.text_ctrl_22.GetLineText(0)]
        
        return spec
 
class TestSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(TestSpecPage, self).__init__(parent, id)

        self.button_1 = wx.Button(self, wx.ID_ANY, _("Browse"))
        self.text_ctrl_24 = wx.TextCtrl(self, wx.ID_ANY, "24")
        self.button_2 = wx.Button(self, wx.ID_ANY, _("Browse"))
        self.text_ctrl_25 = wx.TextCtrl(self, wx.ID_ANY, "25")
        self.text_ctrl_26 = wx.TextCtrl(self, wx.ID_ANY, "26")
        self.text_ctrl_27 = wx.TextCtrl(self, wx.ID_ANY, "27")
        self.combo_box_5 = wx.ComboBox(self, wx.ID_ANY, choices=["5"], style = wx.CB_DROPDOWN | wx.CB_READONLY)

        self.__do_layout()
        self.__do_binds()

    def __do_layout(self):
        grid_sizer_4 = wx.GridBagSizer(0, 0)

        label_14 = wx.StaticText(self, wx.ID_ANY, _("Model select"))
        label_14.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_4.Add(label_14, (1, 1), (1, 10), 0, 0)
        grid_sizer_4.Add(self.combo_box_5, (2, 1), (1, 21), wx.EXPAND, 0)

        label_24 = wx.StaticText(self, wx.ID_ANY, _("Upload image"))
        label_24.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, ""))
        grid_sizer_4.Add(label_24, (3, 1), (1, 10), 0, 0) 
        grid_sizer_4.Add(self.button_1, (5, 1), (1, 4), 0, 0) 
        grid_sizer_4.Add(self.text_ctrl_24, (5, 5), (1, 21), wx.EXPAND | wx.RIGHT, 30) 

        label_25 = wx.StaticText(self, wx.ID_ANY, _("Upload folder")) 
        label_25.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, "")) 
        grid_sizer_4.Add(label_25, (7, 1), (1, 10), 0, 0) 
        grid_sizer_4.Add(self.button_2, (9, 1), (1, 4), 0, 0) 
        grid_sizer_4.Add(self.text_ctrl_25, (9, 5), (1, 21), wx.EXPAND | wx.RIGHT, 30) 

        label_26 = wx.StaticText(self, wx.ID_ANY, _("Number of images use from the file")) 
        label_26.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu")) 
        grid_sizer_4.Add(label_26, (11, 1), (1, 11), 0, 0) 
        grid_sizer_4.Add(self.text_ctrl_26, (13, 1), (1, 25), wx.EXPAND | wx.RIGHT, 30) 

        label_27 = wx.StaticText(self, wx.ID_ANY, _("Leave blank to use all")) 
        label_27.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.LIGHT, 0, "")) 
        grid_sizer_4.Add(label_27, (14, 1), (1, 10), 0, 0) 

        label_28 = wx.StaticText(self, wx.ID_ANY, _("Number of images to show per category")) 
        label_28.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu")) 
        grid_sizer_4.Add(label_28, (15, 1), (1, 14), 0, 0) 
        grid_sizer_4.Add(self.text_ctrl_27, (17, 1), (1, 25), wx.EXPAND | wx.RIGHT, 30) 

        grid_sizer_4.AddGrowableCol(20) 
        self.SetSizer(grid_sizer_4)

    def __do_binds(self):
        pass

    def setTestSpec(self, test_spec):
        self.test_spec = test_spec
        self.combo_box_5.Delete(0)
        for i, model_name in enumerate(test_spec['model_names']):
            self.combo_box_5.Insert(model_name, 0)
        #self.text_ctrl_24.SetValue("")
        #self.text_ctrl_25.SetValue("")

    def setTestSpec_dataset(self):
        self.test_spec['dataset_path']
        self.test_spec['dataset_name']
 
    def getTestSpec(self):
        spec = []
        idx = self.combo_box_5.GetSelection()
        if idx == wx.NOT_FOUND:
            print("select combo box 5")
        else:
            spec += [self.combo_box_5.GetStringSelection()]
        spec += self.test_spec['dataset_path']
        spec += [self.text_ctrl_26.GetLineText(0)]
        spec += [self.text_ctrl_27.GetLineText(0)]
       
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
        print(test_spec_panel, dict)
        self.AddPage(test_spec_panel, _("Test Spec %d"%self.test_spec_count), select=True)
        return test_spec_panel

    #def createTrainSpecPanel(self, parent, id, dict):
    def createTrainSpecPanel(self, parent, id):
        self.train_spec_count += 1
        train_spec_panel = TrainSpecPage(parent, id)
        print(train_spec_panel)
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

#    def getDataSpec(self):
#        pass
#    def setDataSpec(self, data_spec):
#        # name, path, class_n (output_size), image_size (input_shape), ratio of testing, max sample per class
#        data_spec = dict()
#        data_spec['name'] = self.data_tree.GetItemText(dataID)
#        data_spec['path'] = self.data_tree.GetItemData(dataID)
#        
#        return data_spec
#
#    def getTrainSpec(self):
#        pass
#    def setTrainSpec(self):
#        pass
#    def getModelSpec(self):
#        pass
#    def setModelSpec(self):
#        pass
  
    def getSpec(self):
        page = self.GetPage(self.GetSelection())
        if isinstance(page, TrainSpecPage):
            spec = ['train']
            spec += page.getTrainSpec()
        elif isinstance(page, TestSpecPage):
            spec = ['test']
            spec += page.getTestSpec()

        return spec

        spec += page.getModelSpec()
        spec += page.getDatasetSpec()

        if isinstance(page, TrainSpecPage):
            spec = ['train'] 
            # model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
            idx = page.combo_box_2.GetSelection()
            if idx == wx.NOT_FOUND:
                print("select combo box 2")
            else:
                spec += [page.combo_box_2.GetStringSelection()]
            idx = page.combo_box_3.GetSelection()
            if idx == wx.NOT_FOUND:
                print("select combo box 3")
            else:
                spec += [page.combo_box_3.GetStringSelection()]
            idx = page.combo_box_4.GetSelection()
            if idx == wx.NOT_FOUND:
                print("select combo box 4")
            else:
                spec += [page.combo_box_4.GetStringSelection()]
            spec += [page.text_ctrl_15.GetLineText(0)]
            spec += [page.text_ctrl_16.GetLineText(0)]
            spec += [page.text_ctrl_18.GetLineText(0)]
            spec += [page.text_ctrl_20.GetLineText(0)]
            spec += [page.text_ctrl_21.GetLineText(0)]
            spec += [page.text_ctrl_19.GetLineText(0)]
            spec += [page.text_ctrl_22.GetLineText(0)]
            
            return spec
        elif isinstance(page, TestSpecPage):
            spec = ['test']
            idx = page.combo_box_5.GetSelection()
            if idx == wx.NOT_FOUND:
                print("select combo box 5")
            else:
                spec += [page.combo_box_5.GetStringSelection()]
             
#            spec['model'] = 
#            spec['data'] = 
#
            return spec
        else:
            return False
    
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
