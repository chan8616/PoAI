#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.8.2 on Tue Aug 21 13:46:28 2018
#

# This is an automatically generated file.
# Manual changes will be overwritten without warning!

import wx
import sys
sys.path.insert(0, 'gui')
from myframe import MyFrame
#from myframeevent import MyFrameEvent


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, wx.ID_ANY, "")
#        self.frame = MyFrameEvent(frame)
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

# end of class MyApp

if __name__ == "__main__":
    gettext.install("app") # replace with the appropriate catalog name

    app = MyApp(0)
    app.MainLoop()
