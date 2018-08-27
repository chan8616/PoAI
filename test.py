import wx
import wx.aui
import wx.lib.inspection
import random

class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        wx.Frame.__init__(self, *args, **kwds)
        #self.Bind(wx.EVT_CLOSE, self.OnClose)
        menuBar = wx.MenuBar()

        self.nb = wx.lib.agw.aui.auibook.AuiNotebook(self)
        #self.nb = wx.aui.AuiNotebook(self)

        self.new_panel(' ')
        self.new_panel(' ')
        self.new_panel(' ')

        menu = wx.Menu()
    #Create tab
        self.m_load = menu.Append(wx.ID_OPEN,"&Open\tAlt-O", "Create Tab")
        #self.Bind(wx.EVT_MENU, self.new_panel, m_load)
    #Delete tab
        self.m_close = menu.Append(wx.ID_CLOSE,"&Close\tAlt-C", "Delete Tab")
        #self.Bind(wx.EVT_MENU, self.close, m_close)


    #close tab/exit
        self.m_exit = menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Close window and exit program.")
        #self.Bind(wx.EVT_MENU, self.OnClose, m_exit)
        menuBar.Append(menu, "&File")

        self.SetMenuBar(menuBar)
    #display text at bottom of window
        self.statusbar = self.CreateStatusBar()

        #self.nb.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSED, self.close, self.nb)
        print( self.nb.GetSelection())

#create new tab
    def new_panel(self, event):
        nm = str(random.randint(0,999))
        pnl = wx.Panel(self)
        pnl.identifierTag = nm
        self.nb.AddPage(pnl, nm)
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(self.sizer)

class MyFrameEvent(MyFrame):
    def __init__(self, *args, **kwds):
        super(MyFrameEvent, self).__init__(*args, **kwds)
        #self.Bind(wx.EVT_CLOSE, self.OnClose)
        #self.Bind(wx.EVT_MENU, self.new_panel, self.m_load)
        #self.Bind(wx.EVT_MENU, self.close, self.m_close)
        #self.Bind(wx.EVT_MENU, self.OnClose, self.m_exit)
        #self.nb.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSED, self.close, self.nb)

#close window
    def OnClose(self, event):
        print( 'closed',self.nb.GetSelection())
        self.Destroy()

#close tab
    def close(self,event):
        print( 'closed',self.nb.GetSelection())
        self.nb.DeleteAllPages()
        if self.nb.GetSelection() >= 0:
            self.nb.DeletePage(self.nb.GetSelection())

class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, -1, '12_aui_notebook1.py')
        #frame = MyFrameEvent(None, -1, '12_aui_notebook1.py')
        frame = MyFrameEvent(frame)
        frame.Show()
        self.SetTopWindow(frame)
        return 1

if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
