import wx
from queue import Queue
from threading import Thread
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas


class TrainWindow(wx.Frame):
    def __init__(self, parent, title):
        super(TrainWindow, self).__init__(parent, wx.ID_ANY, title=title, size=(520, 480))
        self.progbar_range = 1000
        self.image_width = 480
        self.image_height = 270
        self.InitUI()

    def InitUI(self):
        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox_progbar = wx.BoxSizer(wx.VERTICAL)
        hbox_graph = wx.BoxSizer(wx.HORIZONTAL)

        self.progbar = wx.Gauge(pnl, range=self.progbar_range, size=(self.image_width, 25), style=wx.GA_HORIZONTAL)
        self.msg = wx.StaticText(pnl, label="Test Message")
        #  self.loss_graph = wx.StaticBitmap(pnl, bitmap=wx.NullBitmap,
        #                                    size=(self.image_width, self.image_height), style=wx.GA_HORIZONTAL)
        self.loss_graph = self.generate_loss_graph_canvas(pnl)

        hbox_progbar.Add(self.msg, proportion=1, flag=wx.ALIGN_RIGHT)
        hbox_progbar.Add(self.progbar, proportion=1, flag=wx.ALIGN_CENTRE)
        hbox_graph.Add(self.loss_graph, proportion=1, flag=wx.ALIGN_CENTRE)

        vbox.Add((0, 30))
        vbox.Add(hbox_progbar, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 20))
        vbox.Add(hbox_graph, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 30))
        pnl.SetSizer(vbox)

        #  self.Fit()
        self.SetSize((520, 480))
        self.Centre()
        self.ToggleWindowStyle(wx.FRAME_FLOAT_ON_PARENT)

    def generate_loss_graph_canvas(self, parent):
        #  fig = plt.figure(figsize=(16, 9))
        fig = plt.figure(figsize=(8, 4.5))
        self.ax = ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(xmin=0., xmax=None, auto=True)
        ax.set_ylim(ymin=0., ymax=None, auto=True)
        ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Graph')

        self.batch_losses_plot, = ax.plot([], [], 'g--', label='Train Batch', alpha=0.3)
        self.epoch_losses_plot, = ax.plot([], [], 'c.-', label='Train Epoch')
        self.epoch_val_losses_plot, = ax.plot([], [], 'b.-', label='Validation Epoch')

        self.legend = ax.legend()

        return FigureCanvas(parent, -1, fig)

    def update_msg(self, msg):
        #  self.msg.SetLabelText(msg)
        wx.CallAfter(self.msg.SetLabelText, msg)

    def update_gauge(self, ratio):
        self.progbar.SetValue(int(ratio * self.progbar_range))
        #  wx.CallAfter(self.progbar.SetValue, int(ratio * self.progbar_range))

    def update_loss_graph(self, batch_losses, epoch_losses, epoch_val_losses):
        print(bool(batch_losses), bool(epoch_losses), bool(epoch_val_losses))
        self.ax.clear()
        self.batch_losses_plot, = self.ax.plot(
                *(([], []) if not batch_losses else zip(*batch_losses)),
                'g--', label='Train Batch', alpha=0.3)
        epochs = range(1, len(epoch_losses) + 1, 1)
        self.epoch_losses_plot, = self.ax.plot(
                *(([], []) if not epoch_losses else (epochs, epoch_losses)),
                'c.-', label='Train Epoch')
        self.epoch_val_losses_plot, = self.ax.plot(
                *(([], []) if not epoch_val_losses else (epochs, epoch_val_losses)),
                'b.-', label='Validation Epoch')
        if batch_losses or epoch_losses or epoch_val_losses:
            self.ax.legend()
            #  self.loss_graph.draw()
            wx.CallAfter(self.loss_graph.draw, ())


class TrainWindowManager(object):
    def __init__(self, parent, stream: Queue):
        self.train_window = TrainWindow(parent, title="Train Window")
        self.stream = stream

        self.batch_losses = []
        self.batch_acces = []
        self.epoch_losses = []
        self.epoch_acces = []
        self.epoch_val_losses = []
        self.epoch_val_acces = []

        self.loss_graph_buf = None
        self.acc_graph_buf = None

        self.cur_step_text = "Starting..."
        self.msg_text = ""

    def main_loop(self):
        self.train_window.Show()
        self.train_window.update_msg(self.cur_step_text)

        batch_print_target_ratio = 0.0
        batch_print_ratio_steps = 0.1
        current_epoch = 0

        while True:
            print_graph = False

            data = self.stream.get(block=True)
            if data == 'end':
                self.train_window.update_msg('End')
                break

            data_head, data_body, data_msg = data
            if data_head == 'batch':
                current_batch_num, total_batch_num, batch_loss, batch_acc = data_body
                batch_progress_ratio = current_batch_num / total_batch_num
                if batch_progress_ratio + current_epoch >= batch_print_target_ratio:
                    print_graph = True
                    batch_print_target_ratio += batch_print_ratio_steps
                self.batch_losses.append((batch_progress_ratio + current_epoch, batch_loss))
                self.train_window.update_gauge(batch_progress_ratio)
            elif data_head == 'epoch':
                current_epoch_num, epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc = data_body
                current_epoch = current_epoch_num + 1
                self.epoch_losses.append(epoch_loss)
                self.epoch_acces.append(epoch_acc)
                self.epoch_val_losses.append(epoch_val_loss)
                self.epoch_val_acces.append(epoch_val_acc)
            else:
                print_graph = True
                self.cur_step_text = data_head

            if data_msg is not None:
                self.msg_text = data_msg

            self.train_window.update_msg(self.msg_text + " " + self.cur_step_text)
            if print_graph:
                self.train_window.update_loss_graph(self.batch_losses,
                                                    self.epoch_losses,
                                                    self.epoch_val_losses,)

        #  plt.close(fig)
        self.train_window.Close()


class TrainThread(Thread):
    def __init__(self, train_function, config, stream: Queue):
        super(TrainThread, self).__init__()
        self.train_function = train_function
        config_list = list(config)
        config_list.append(stream)
        self.config = tuple(config_list)

    def run(self):
        self.train_function(self.config)


if __name__ == "__main__":
    app = wx.App(False)
    frame = TrainWindow(None, 'TrainWindows')
    frame.update_gauge(0.42)
    frame.update_msg("Changed!")
    frame.Show()
    app.MainLoop()
