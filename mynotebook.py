import wx
import wx.lib.inspection

class DataSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(DataSpecPage, self).__init__(parent, id)

        self.text_ctrl_2 = wx.TextCtrl(self, wx.ID_ANY, "2", style=wx.TE_READONLY)
        self.text_ctrl_3 = wx.TextCtrl(self, wx.ID_ANY, "3", style=wx.TE_READONLY)
        self.text_ctrl_4 = wx.TextCtrl(self, wx.ID_ANY, "4", style=wx.TE_READONLY)
        self.text_ctrl_5 = wx.TextCtrl(self, wx.ID_ANY, "5", style=wx.TE_READONLY)
        self.text_ctrl_6 = wx.TextCtrl(self, wx.ID_ANY, "6", style=wx.TE_READONLY)
        self.combo_box_1 = wx.ComboBox(self, wx.ID_ANY, choices=["combo_box_1"], style=wx.CB_DROPDOWN)
        self.text_ctrl_8 = wx.TextCtrl(self, wx.ID_ANY, "8")
        self.text_ctrl_7 = wx.TextCtrl(self, wx.ID_ANY, "7")

        self.__do_layout()

    def __init__(self, parent, id, dict):
        super(DataSpecPage, self).__init__(parent, id)

        self.text_ctrl_2 = wx.TextCtrl(self, wx.ID_ANY, dict['path'] if 'path' in dict else "2", style=wx.TE_READONLY)
        self.text_ctrl_3 = wx.TextCtrl(self, wx.ID_ANY, "3", style=wx.TE_READONLY)
        self.text_ctrl_4 = wx.TextCtrl(self, wx.ID_ANY, "4", style=wx.TE_READONLY)
        self.text_ctrl_5 = wx.TextCtrl(self, wx.ID_ANY, "5", style=wx.TE_READONLY)
        self.text_ctrl_6 = wx.TextCtrl(self, wx.ID_ANY, "6", style=wx.TE_READONLY)
        self.combo_box_1 = wx.ComboBox(self, wx.ID_ANY, choices=["combo_box_1"], style=wx.CB_DROPDOWN)
        self.text_ctrl_8 = wx.TextCtrl(self, wx.ID_ANY, "8")
        self.text_ctrl_7 = wx.TextCtrl(self, wx.ID_ANY, "7")

        self.__do_layout()

    def __do_layout(self):
        grid_sizer_1 = wx.GridBagSizer(5, 5)
        label_1 = wx.StaticText(self, wx.ID_ANY, _("Dataset Path"), style=wx.ALIGN_LEFT)
        label_1.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_1, (1, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_2, (1, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_2 = wx.StaticText(self, wx.ID_ANY, _("Number of Classes"), style=wx.ALIGN_LEFT)
        label_2.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_2, (2, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_3, (2, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_3 = wx.StaticText(self, wx.ID_ANY, _("Number of Images"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_3.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_3, (3, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_4, (3, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_4 = wx.StaticText(self, wx.ID_ANY, _("Image Size"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_4.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_4, (4, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_5, (4, 8), (1, 13), wx.EXPAND | wx.RIGHT, 0)
        label_7 = wx.StaticText(self, wx.ID_ANY, _("X"))
        grid_sizer_1.Add(label_7, (4, 21), (1, 1), wx.ALIGN_CENTER, 0)
        grid_sizer_1.Add(self.text_ctrl_6, (4, 22), (1, 13), wx.EXPAND | wx.RIGHT, 20)
        label_5 = wx.StaticText(self, wx.ID_ANY, _("Image Type"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_5.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_5, (5, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.combo_box_1, (5, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_6 = wx.StaticText(self, wx.ID_ANY, _("% of Testing"), style=wx.ALIGN_CENTER | wx.ALIGN_LEFT)
        label_6.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_6, (6, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_8, (6, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_8 = wx.StaticText(self, wx.ID_ANY, _("Maximum Samples per Class"), style=wx.ALIGN_LEFT)
        label_8.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_1.Add(label_8, (7, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.text_ctrl_7, (7, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
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
        self.text_ctrl_13 = wx.TextCtrl(self, wx.ID_ANY, "13")
        self.text_ctrl_12 = wx.TextCtrl(self, wx.ID_ANY, "12")

        self.__do_layout()

    def __init__(self, parent, id, dict):
        super(ModelSpecPage, self).__init__(parent, id)

        self.text_ctrl_9 = wx.TextCtrl(self, wx.ID_ANY, dict['type'] if 'type' in dict else "9")
        self.text_ctrl_10 = wx.TextCtrl(self, wx.ID_ANY, dict['n_layer'] if 'n_layer' in dict else "10")
        self.text_ctrl_11 = wx.TextCtrl(self, wx.ID_ANY, dict['input_size'] if 'image_size' in dict else "11")
        self.text_ctrl_13 = wx.TextCtrl(self, wx.ID_ANY, "13")
        self.text_ctrl_12 = wx.TextCtrl(self, wx.ID_ANY, dict['output_size'] if 'output_size' in dict else "12")

        self.__do_layout()
    
    def __do_layout(self):
        grid_sizer_2 = wx.GridBagSizer(5, 5)
        label_9 = wx.StaticText(self, wx.ID_ANY, _("Network Type"), style=wx.ALIGN_LEFT)
        label_9.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_9, (1, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_9, (1, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_10 = wx.StaticText(self, wx.ID_ANY, _("Number of Layers"), style=wx.ALIGN_LEFT)
        label_10.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_10, (2, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_10, (2, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        label_11 = wx.StaticText(self, wx.ID_ANY, _("Image Size (Input)"), style=wx.ALIGN_LEFT)
        label_11.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_11, (3, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_11, (3, 8), (1, 13), wx.EXPAND, 0)
        label_13 = wx.StaticText(self, wx.ID_ANY, _("X"), style=wx.ALIGN_CENTER)
        grid_sizer_2.Add(label_13, (3, 21), (1, 1), wx.ALIGN_CENTER, 0)
        grid_sizer_2.Add(self.text_ctrl_13, (3, 22), (1, 13), wx.EXPAND | wx.RIGHT, 20)
        label_12 = wx.StaticText(self, wx.ID_ANY, _("Number of Classes / Demension (Output)"))
        label_12.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_2.Add(label_12, (4, 1), (1, 6), wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.text_ctrl_12, (4, 8), (1, 27), wx.EXPAND | wx.RIGHT, 20)
        self.SetSizer(grid_sizer_2)
#        grid_sizer_2.AddGrowableRow(18)
#        grid_sizer_2.AddGrowableCol(8)
#        grid_sizer_2.AddGrowableCol(22)




class TrainSpecPage(wx.Panel):
    def __init__(self, parent, id):
        super(TrainSpecPage, self).__init__(parent, id)

        self.text_ctrl_14 = wx.TextCtrl(self, wx.ID_ANY, "14")
        self.text_ctrl_19 = wx.TextCtrl(self, wx.ID_ANY, "19")
        self.text_ctrl_15 = wx.TextCtrl(self, wx.ID_ANY, "15")
        self.text_ctrl_20 = wx.TextCtrl(self, wx.ID_ANY, "20")
        self.text_ctrl_16 = wx.TextCtrl(self, wx.ID_ANY, "26")
        self.text_ctrl_21 = wx.TextCtrl(self, wx.ID_ANY, "21")
        self.text_ctrl_17 = wx.TextCtrl(self, wx.ID_ANY, "17")
        self.text_ctrl_22 = wx.TextCtrl(self, wx.ID_ANY, "22")
        self.text_ctrl_18 = wx.TextCtrl(self, wx.ID_ANY, "18")
        self.text_ctrl_23 = wx.TextCtrl(self, wx.ID_ANY, "23")

        self.__do_layout()
    
    def __init__(self, parent, id, dict):
        super(TrainSpecPage, self).__init__(parent, id)

        self.text_ctrl_14 = wx.TextCtrl(self, wx.ID_ANY, "14")
        self.text_ctrl_19 = wx.TextCtrl(self, wx.ID_ANY, str(dict['validation interval']) if 'validation interval' in dict else "19")
        self.text_ctrl_15 = wx.TextCtrl(self, wx.ID_ANY, "15")
        self.text_ctrl_20 = wx.TextCtrl(self, wx.ID_ANY, dict['optimizer'] if 'optimizer' in dict else "20")
        self.text_ctrl_16 = wx.TextCtrl(self, wx.ID_ANY, str(dict['max_iter']) if 'max_iter' in dict else "26")
        self.text_ctrl_21 = wx.TextCtrl(self, wx.ID_ANY, str(dict['lr']) if 'lr' in dict else "21")
        self.text_ctrl_17 = wx.TextCtrl(self, wx.ID_ANY, "17")
        self.text_ctrl_22 = wx.TextCtrl(self, wx.ID_ANY, str(dict['seed']) if 'seed' in dict else "22")
        self.text_ctrl_18 = wx.TextCtrl(self, wx.ID_ANY, str(dict['batch_size']) if 'batch_size' in dict else "18")
        self.text_ctrl_23 = wx.TextCtrl(self, wx.ID_ANY, "23")
        
        self.__do_layout()

    def __do_layout(self):
        grid_sizer_3 = wx.GridBagSizer(0, 0)
        label_14 = wx.StaticText(self, wx.ID_ANY, _("Model select"))
        label_14.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_14, (1, 1), (1, 10), 0, 0)
        label_19 = wx.StaticText(self, wx.ID_ANY, _("Sanpshot interval(in epochs)"))
        label_19.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_19, (1, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_14, (2, 1), (1, 18), wx.EXPAND, 0)
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
        label_21 = wx.StaticText(self, wx.ID_ANY, _("Base Learning rate"))
        label_21.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_21, (7, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_16, (8, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_21, (8, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30) 
        label_17 = wx.StaticText(self, wx.ID_ANY, _("Dataset name"))
        label_17.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_17, (10, 1), (1, 10), 0, 0)
        label_22 = wx.StaticText(self, wx.ID_ANY, _("Random seed"))
        label_22.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_22, (10, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_17, (11, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_22, (11, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30) 
        label_18 = wx.StaticText(self, wx.ID_ANY, _("Batch size"))
        label_18.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_18, (13, 1), (1, 10), 0, 0)
        label_23 = wx.StaticText(self, wx.ID_ANY, _("Select which GPU[s] you would like to use"))
        label_23.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu"))
        grid_sizer_3.Add(label_23, (13, 21), (1, 10), 0, 0)
        grid_sizer_3.Add(self.text_ctrl_18, (14, 1), (1, 18), wx.EXPAND, 0)
        grid_sizer_3.Add(self.text_ctrl_23, (14, 21), (1, 18), wx.EXPAND | wx.RIGHT, 30)
        self.SetSizer(grid_sizer_3)

class MyNotebook(wx.lib.agw.aui.auibook.AuiNotebook):
    def __init__(self, *args, **kwds):
		# Auinotebook
        super(MyNotebook, self).__init__(*args, **kwds)
        
        self.data_spec_count = 0 
        self.model_spec_count = 0 
        self.train_spec_count = 0 
        self.Testmodelsindleimage = wx.Panel(self, wx.ID_ANY)
        self.Testmodelimagefolder = wx.Panel(self, wx.ID_ANY)
        self.button_1 = wx.Button(self.Testmodelsindleimage, wx.ID_ANY, _("Browse"))
        self.text_ctrl_24 = wx.TextCtrl(self.Testmodelsindleimage, wx.ID_ANY, "24")
        self.button_2 = wx.Button(self.Testmodelimagefolder, wx.ID_ANY, _("Browse"))
        self.text_ctrl_25 = wx.TextCtrl(self.Testmodelimagefolder, wx.ID_ANY, "25")
        self.text_ctrl_26 = wx.TextCtrl(self.Testmodelimagefolder, wx.ID_ANY, "26")
        self.text_ctrl_27 = wx.TextCtrl(self.Testmodelimagefolder, wx.ID_ANY, "27")

        self.__do_layout()
        # Auinotebook end

    def __do_layout(self):
        grid_sizer_5 = wx.GridBagSizer(0, 0)
        grid_sizer_4 = wx.GridBagSizer(0, 0)

        label_24 = wx.StaticText(self.Testmodelsindleimage, wx.ID_ANY, _("Upload image"))
        label_24.SetFont(wx.Font(15, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, ""))
        grid_sizer_4.Add(label_24, (1, 1), (1, 8), 0, 0) 
        grid_sizer_4.Add(self.button_1, (3, 1), (1, 4), 0, 0) 
        grid_sizer_4.Add(self.text_ctrl_24, (3, 5), (1, 21), wx.EXPAND | wx.RIGHT, 30) 
        self.Testmodelsindleimage.SetSizer(grid_sizer_4) 
#        grid_sizer_4.AddGrowableRow(18) 
        grid_sizer_4.AddGrowableCol(25) 
#        grid_sizer_4.AddGrowableCol(26) 
#        grid_sizer_4.AddGrowableCol(27) 
        label_25 = wx.StaticText(self.Testmodelimagefolder, wx.ID_ANY, _("Upload folder")) 
        label_25.SetFont(wx.Font(15, wx.DEFAULT, wx.NORMAL, wx.BOLD, 0, "")) 
        grid_sizer_5.Add(label_25, (1, 1), (1, 8), 0, 0) 
        grid_sizer_5.Add(self.button_2, (3, 1), (1, 4), 0, 0) 
        grid_sizer_5.Add(self.text_ctrl_25, (3, 5), (1, 21), wx.EXPAND | wx.RIGHT, 30) 
        label_26 = wx.StaticText(self.Testmodelimagefolder, wx.ID_ANY, _("Number of images use from the file")) 
        label_26.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu")) 
        grid_sizer_5.Add(label_26, (5, 1), (1, 11), 0, 0) 
        grid_sizer_5.Add(self.text_ctrl_26, (6, 1), (1, 25), wx.EXPAND | wx.RIGHT, 30) 
        label_27 = wx.StaticText(self.Testmodelimagefolder, wx.ID_ANY, _("Leave blank to use all")) 
        label_27.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.LIGHT, 0, "")) 
        grid_sizer_5.Add(label_27, (7, 1), (1, 8), 0, 0) 
        label_28 = wx.StaticText(self.Testmodelimagefolder, wx.ID_ANY, _("Number of images to show per category")) 
        label_28.SetFont(wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Ubuntu")) 
        grid_sizer_5.Add(label_28, (9, 1), (1, 14), 0, 0) 
        grid_sizer_5.Add(self.text_ctrl_27, (10, 1), (1, 25), wx.EXPAND | wx.RIGHT, 30) 
        self.Testmodelimagefolder.SetSizer(grid_sizer_5) 
#        grid_sizer_5.AddGrowableRow(18) 
        grid_sizer_5.AddGrowableCol(25) 
#        grid_sizer_5.AddGrowableCol(26) 
#        grid_sizer_5.AddGrowableCol(27) 
        #self.notebook_1.AddPage(self.DataSpec, _("Dataset Spec")) 
#        for i in range(len(self.panel_1_list)): 
#            self.notebook_1.AddPage(self.panel_1_list[i], _("Dataset Spec")) 
        #self.notebook_1.AddPage(self.ModelSpec, _("Model Spec")) 
        #self.notebook_1.AddPage(self.Makeamodel, _("Make a model")) 
        self.AddPage(self.Testmodelsindleimage, _("Test model(sindle image)")) 
        self.AddPage(self.Testmodelimagefolder, _("Test model(image folder)")) 
        self.Layout()

    def createTrainSpecPanel(self, parent, id, dict):
        print('createTrainSpecPanel')
        self.train_spec_count += 1
        train_spec_panel = TrainSpecPage(parent, id, dict)
        print(train_spec_panel)
        self.AddPage(train_spec_panel, _("Train Spec %d"%self.train_spec_count), select=True)
        return train_spec_panel

    def createModelSpecPanel(self, parent, id, dict):
        self.model_spec_count += 1
        model_spec_panel = ModelSpecPage(parent, id, dict)
        self.AddPage(model_spec_panel, _("Model Spec %d"%self.model_spec_count), select=True)
        return model_spec_panel

    def createDataSpecPanel(self, parent, id, dict):
        self.data_spec_count += 1
        data_spec_panel = DataSpecPage(parent, id, dict)
        self.AddPage(data_spec_panel, _("Data Spec %d"%self.data_spec_count), select=True)
        return data_spec_panel


    def OnTabClicked(self, event):
        super(MyNotebook, self).OnTabClicked(event)
        #print('tab clicked', event, event.GetNotifyEvent())
        #event.Skip()
