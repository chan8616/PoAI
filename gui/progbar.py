import wx
from queue import Queue
from threading import Thread
import io
import matplotlib.pyplot as plt


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
        self.loss_graph = wx.StaticBitmap(pnl, bitmap=wx.Bitmap(self.image_width, self.image_height),
                                          size=(self.image_width, self.image_height), style=wx.GA_HORIZONTAL)

        hbox_progbar.Add(self.msg, proportion=1, flag=wx.ALIGN_RIGHT)
        hbox_progbar.Add(self.progbar, proportion=1, flag=wx.ALIGN_CENTRE)
        hbox_graph.Add(self.loss_graph, proportion=1, flag=wx.ALIGN_CENTRE)

        vbox.Add((0, 30))
        vbox.Add(hbox_progbar, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 20))
        vbox.Add(hbox_graph, flag=wx.ALIGN_CENTRE)
        vbox.Add((0, 30))
        pnl.SetSizer(vbox)

        self.SetSize((520, 480))
        self.Centre()
        self.ToggleWindowStyle(wx.FRAME_FLOAT_ON_PARENT)

    def update_msg(self, msg):
        self.msg.SetLabelText(msg)

    def update_gauge(self, ratio):
        self.progbar.SetValue(int(ratio * self.progbar_range))

    def update_loss_graph(self, img_buf):
        image = wx.Image(img_buf, wx.BITMAP_TYPE_ANY)
        image = image.Scale(self.image_width, self.image_height, wx.IMAGE_QUALITY_HIGH)
        self.loss_graph.SetBitmap(wx.Bitmap(image))


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

        self.cur_step_text = "Epoch 0"
        self.msg_text = ""

    def main_loop(self):
        self.train_window.Show()
        self.train_window.update_msg("Starting...")
        while True:
            data = self.stream.get(block=True)
            if data == 'end':
                break

            data_head, data_body, data_msg = data
            if data_head == 'batch':
                current_batch_num, total_batch_num, batch_loss, batch_acc = data_body
                batch_progress_ratio = current_batch_num / total_batch_num
                self.batch_losses.append(batch_loss)
                self.train_window.update_gauge(batch_progress_ratio)
            elif data_head == 'epoch':
                current_epoch_num, epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc = data_body
                current_batch = len(self.batch_losses)
                self.cur_step_text = f"Epoch {current_epoch_num}"
                self.epoch_losses.append((current_batch, epoch_loss))
                self.epoch_acces.append((current_batch, epoch_acc))
                self.epoch_val_losses.append((current_batch, epoch_val_loss))
                self.epoch_val_acces.append((current_batch, epoch_val_acc))
            else:
                self.cur_step_text = data_head

            if data_msg is not None:
                self.msg_text = data_msg

            self.train_window.update_msg(self.msg_text + " " + self.cur_step_text)
            self.train_window.update_loss_graph(self.generate_loss_graph_img())

        self.train_window.Close()

    def generate_loss_graph_img(self):
        batches = range(1, len(self.batch_losses)+1, 1)
        batches_losses = self.batch_losses
        epoch_losses = self.epoch_losses
        val_losses = self.epoch_val_losses

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(xmin=0., xmax=None, auto=True)
        ax.set_ylim(ymin=0., ymax=None, auto=True)
        ax.set(xlabel='Batch', ylabel='Loss', title='Loss Graph')
        if batches_losses:
            ax.plot(batches, batches_losses, 'g--', label='Batch')
        if epoch_losses:
            ax.plot(*zip(*epoch_losses), 'c.-', label='Epoch')
        if val_losses:
            ax.plot(*zip(*val_losses), 'bo-', label='Validation')
        if batches_losses or epoch_losses or val_losses:
            ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf


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
