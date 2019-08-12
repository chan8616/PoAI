import wx
import sys
import time
from train import build, load_dataset, train
from threading import Thread

EVT_RESULT_ID = wx.NewId()

def EVT_RESULT(win, func):
    """"""
    win.Connect(-1, -1, EVT_RESULT_ID, func)

class ResultEvent(wx.PyEvent):
    """"""
    def __init__(self, data):
        """"""
        super(ResultEvent, self).__init__()
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

class TestThread(Thread):
    """"""
    def __init__(self, wxObject):
        """"""
        super(TestThread, self).__init__()
        self.wxObject = wxObject
        self.start()

    def run(self):
        """"""
        model = build()
        dataset = load_dataset('train')
        # model.compile('adam', 'sparse_categorical_crossentropy')
        # model.fit(*dataset, callbacks=
        # sys.stdout = self.wxObject
        train(model, dataset)
        # for i in range(10):
        #     time.sleep(1)
        #     t = (i+1)*10
        #     print(t)
        #     wx.PostEvent(self.wxObject, ResultEvent(t))
        # time.sleep(1)
        # wx.PostEvent(self.wxObject, ResultEvent("Thread finished!"))


class Mywin(wx.Frame):
    """"""
    def __init__(self, parent, title):
        """"""
        super(Mywin, self).__init__(parent, wx.ID_ANY, title=title, size=(300, 200))
        self.InitUI()

    def InitUI(self):
        """"""
        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.gauge = wx.Gauge(pnl, range=100, size=(250, 25), style=wx.GA_HORIZONTAL)
        self.btn1 = wx.Button(pnl, label="Start")
        # self.Bind(wx.EVT_BUTTON, self.OnStart, self.btn1)
        self.btn1.Bind(wx.EVT_BUTTON, self.OnStart)

        hbox1.Add(self.gauge, proportion=1, flag=wx.ALIGN_CENTRE)
        hbox2.Add(self.btn1, proportion=1, flag=wx.RIGHT, border=10)

        vbox.Add((0, 30))
        vbox.Add(hbox1, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 20))
        vbox.Add(hbox2, proportion=1, flag=wx.ALIGN_CENTRE)
        pnl.SetSizer(vbox)

        EVT_RESULT(self, self.updateDisplay)
        # print(wx.EVT_BUTTON)
        # self.Bind(EVT_RESULT_ID, self.updateDisplay, self.gauge)

        self.SetSize((300, 200))
        self.Centre()
        # self.Show(True)

    def OnStart(self, event):
        """"""
        TestThread(self)
        # self.gauge.SetLabel("Thread Start!")
        print("Thread Start!")
        btn = event.GetEventObject()
        btn.Disable()
        # while True:
        #     time.sleep(1)
        #     self.count += 1
        #     self.gauge.SetValue(self.count)

        #     if self.count >= 20:
        #         print("end")
        #         return

    def updateDisplay(self, event):
        """"""
        print('update Display')
        t = event.data
        if isinstance(t, int):
            self.gauge.SetValue(t)
        else:
            print("end")
            self.btn1.Enable()

if __name__ == "__main__":
    app = wx.App()
    Mywin(None, 'wx.Gauge').Show()
    app.MainLoop()
