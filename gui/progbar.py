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
