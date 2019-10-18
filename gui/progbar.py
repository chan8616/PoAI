import wx
from queue import Queue
from threading import Thread


class TrainWindow(wx.Frame):
    def __init__(self, parent, title, stream: Queue):
        super(TrainWindow, self).__init__(parent, wx.ID_ANY, title=title, size=(300, 200))
        self.stream = stream
        self.progbar_range = 1000
        self.InitUI()

    def InitUI(self):
        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.progbar = wx.Gauge(pnl, range=self.progbar_range, size=(250, 25), style=wx.GA_HORIZONTAL)

        hbox1.Add(self.progbar, proportion=1, flag=wx.ALIGN_CENTRE)

        vbox.Add((0, 30))
        vbox.Add(hbox1, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 30))
        pnl.SetSizer(vbox)

        self.SetSize((300, 200))
        self.Centre()
        self.ToggleWindowStyle(wx.FRAME_FLOAT_ON_PARENT)

    def main_loop(self):
        self.Show()
        while True:
            data = self.stream.get(block=True)
            if data == 'end':
                break
            else:
                current, total, msg = data
                percentage = current / total
                self.update_gauge(percentage)

    def update_gauge(self, percentage):
        self.progbar.SetValue(int(percentage * self.progbar_range))


class TrainThread(Thread):
    def __init__(self, train_function, config, stream: Queue):
        super(TrainThread, self).__init__()
        self.train_function = train_function
        config_list = list(config)
        config_list.append(stream)
        self.config = tuple(config_list)

    def run(self):
        self.train_function(self.config)

import io
import wx
import numpy as np
import matplotlib.pyplot as plt


class MainPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.SetSize((512, 384))
        self.SetBackgroundColour('white')

        #Generate Sample Graph
        t = np.arange(0.0, 2.0, 0.01)
        s = 10 * np.power(np.e, -t + np.random.randn())
        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='epoch', ylabel='loss',
               title='Loss Graph')
        ax.grid()

        #Save into Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        self.Image = wx.Image(buf, wx.BITMAP_TYPE_ANY)
        self.Image = wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap(self.Image))
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.Image, 1, wx.ALIGN_CENTRE)
        self.SetSizer(self.sizer)


class MyForm(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY,
            "Graph to Image Test", size=(1024, 768))
        self.panel = MainPanel(self)


if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()