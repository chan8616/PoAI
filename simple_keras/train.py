from keras.models import Sequential
from keras.layers import *

from keras.callbacks import LambdaCallback, ProgbarLogger
from callback import MyProgbarLogger

from keras.datasets import cifar10 as data

def build():
    model = Sequential()
    model.add(Reshape([32*32*3], input_shape=(32, 32, 3)))
    model.add(Dense(10))
    return model

def load_dataset(mode):
    train, test = data.load_data()
    if mode == 'train':
        return train
    elif mode == 'validation':
        return test
    elif mode == 'test':
        return test[:1]

def train(model, dataset, callbacks=[]):
    model.compile('adam', 'sparse_categorical_crossentropy')
    model.fit(*dataset, callbacks=callbacks)


if __name__ == '__main__':
    train(build(), load_dataset('train'))

# def lambda_callback(func):
#     return LambdaCallback(

