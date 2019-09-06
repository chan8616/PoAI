import wx
from threading import Thread
from model.train import train
from model.build import build
from model.test import test

from gui.util.progbar_util import EVT_RESULT, MyProgbarLogger


class TrainWindow(wx.Frame):
    def __init__(self, parent, title):
        super(TrainWindow, self).__init__(parent, wx.ID_ANY, title=title, size=(300, 200))
        self.InitUI()

    def InitUI(self):
        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.gauge = wx.Gauge(pnl, range=100, size=(250, 25), style=wx.GA_HORIZONTAL)

        hbox1.Add(self.gauge, proportion=1, flag=wx.ALIGN_CENTRE)

        vbox.Add((0, 30))
        vbox.Add(hbox1, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 30))
        pnl.SetSizer(vbox)

        EVT_RESULT(self, self.updateDisplay)

        self.SetSize((300, 200))
        self.Centre()

    def train(self, model_cmd, args1, args2):
        self.Show()
        thread = TrainThread(self, model_cmd, args1, args2)
        thread.setDaemon(True)
        thread.start()
        thread.join()
        self.Close()

    def updateDisplay(self, event):
        t = event.data
        if isinstance(t, float):
            self.gauge.SetValue(int(t * 100))


class TrainThread(Thread):
    def __init__(self, wxObject, model_cmd, args1, args2):
        super(TrainThread, self).__init__()
        self.wxObject = wxObject
        self.model_cmd = model_cmd
        _, _, _, callbacks, _ = args1
        callbacks.append(MyProgbarLogger(self.wxObject))
        self.args1 = args1
        self.args2 = args2

    def run(self):
        train(self.model_cmd, self.args1, self.args2)


class TestWindow(wx.Frame):
    def __init__(self, parent, title):
        super(TestWindow, self).__init__(parent, wx.ID_ANY, title=title, size=(300, 200))
        self.InitUI()

    def InitUI(self):
        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.gauge = wx.Gauge(pnl, range=100, size=(250, 25), style=wx.GA_HORIZONTAL)

        hbox1.Add(self.gauge, proportion=1, flag=wx.ALIGN_CENTRE)

        vbox.Add((0, 30))
        vbox.Add(hbox1, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 30))
        pnl.SetSizer(vbox)

        EVT_RESULT(self, self.updateDisplay)

        self.SetSize((300, 200))
        self.Centre()

    def train(self, model_cmd, args1, args2):
        self.Show()
        thread = TestThread(self, model_cmd, args1, args2)
        thread.setDaemon(True)
        thread.start()
        thread.join()
        self.Close()

    def updateDisplay(self, event):
        t = event.data
        if isinstance(t, float):
            self.gauge.SetValue(int(t * 100))


class TestThread(Thread):
    def __init__(self, wxObject, model_cmd, args1, args2):
        super(TestThread, self).__init__()
        self.wxObject = wxObject
        self.model_cmd = model_cmd
        _, _, _, callbacks, _ = args1
        callbacks.append(MyProgbarLogger(self.wxObject))
        self.args1 = args1
        self.args2 = args2

    def run(self):
        test(self.model_cmd, self.args1, self.args2)


class BuildWindow(wx.Frame):
    def __init__(self, parent, title):
        super(BuildWindow, self).__init__(parent, wx.ID_ANY, title=title, size=(300, 200))
        self.InitUI()

    def InitUI(self):
        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.gauge = wx.Gauge(pnl, range=100, size=(250, 25), style=wx.GA_HORIZONTAL)

        hbox1.Add(self.gauge, proportion=1, flag=wx.ALIGN_CENTRE)

        vbox.Add((0, 30))
        vbox.Add(hbox1, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 30))
        pnl.SetSizer(vbox)

        EVT_RESULT(self, self.updateDisplay)

        self.SetSize((300, 200))
        self.Centre()

    def train(self, model_cmd, args):
        self.Show()
        thread = BuildThread(self, model_cmd, args)
        thread.setDaemon(True)
        thread.start()
        thread.join()
        self.Close()

    def updateDisplay(self, event):
        t = event.data
        if isinstance(t, float):
            self.gauge.SetValue(int(t * 100))


class BuildThread(Thread):
    def __init__(self, wxObject, model_cmd, args):
        super(BuildThread, self).__init__()
        self.wxObject = wxObject
        self.model_cmd = model_cmd
        _, _, _, callbacks, _ = args
        callbacks.append(MyProgbarLogger(self.wxObject))
        self.args = args

    def run(self):
        build(self.model_cmd, self.args)