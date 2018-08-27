import wx
import wx.lib.inspection

class MyNotebook(wx.lib.agw.aui.auibook.AuiNotebook):
    def __init__(self, *args, **kwds):
		# Auinotebook
        super(MyNotebook, self).__init__(*args, **kwds)
        
        self.panel_1_list = []#[self.createDataSpecPanel(self)]
        self.panel_1_count = 0 
        self.panel_2_list = []#[self.createModelSpecPanel(self)]
        self.panel_2_count = 0 
        self.panel_3_list = []#[self.createTrainSpecPanel(self)]
        self.panel_3_count = 0 
        #self.panel_4_list = []
        #self.ModelSpec = wx.Panel(self.notebook_1, wx.ID_ANY)
        #self.Makeamodel = wx.Panel(self.notebook_1, wx.ID_ANY)
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
        for i in range(len(self.panel_1_list)): 
            self.notebook_1.AddPage(self.panel_1_list[i], _("Dataset Spec")) 
        #self.notebook_1.AddPage(self.ModelSpec, _("Model Spec")) 
        #self.notebook_1.AddPage(self.Makeamodel, _("Make a model")) 
        self.AddPage(self.Testmodelsindleimage, _("Test model(sindle image)")) 
        self.AddPage(self.Testmodelimagefolder, _("Test model(image folder)")) 
        self.Layout()

    def OnTabClicked(self, event):
        super(MyNotebook, self).OnTabClicked(event)
        #print('tab clicked', event, event.GetNotifyEvent())
        #event.Skip()

