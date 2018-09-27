from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import CuDNNLSTM as lstm, Input
from .model import NET

class LSTM(NET):

    def __init__(self, **kargs):
        kargs['model'] = 'lstm'
        kargs['init'] = [None, None]
        super(LSTM, self).__init__(**kargs)
        self.time_stamp = kargs['time_stamp']
        self.input_dim = kargs['input_dim']
    def build_model(self, conf):
        input = Input(shape=(self.time_stamp, self.input_dim,))


    def test(self, x, y=None):
        if y is not None:
            return self.accuracy(x, y)
        else:
            return self.predict(x)
