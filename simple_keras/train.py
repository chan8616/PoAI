from keras.models import Sequential
from keras import layers as KL

from keras.callbacks import LambdaCallback, ProgbarLogger
from callback import MyProgbarLogger

from keras.datasets import cifar10 as data


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def build():
    model = Sequential()
    #  model.add(Reshape([32*32*3], input_shape=(32, 32, 3)))
    #  model.add(Reshape([32*32*3], input_shape=(32, 32, 3)))
    model.add(KL.ZeroPadding2D((3, 3), input_shape=(32, 32, 3)))
    model.add(KL.Conv2D(64, (7, 7), strides=(2, 2), padding="same"))
    #  model.add(BatchNorm(name='bn_conv1')(x, training=train_bn)
    #  model.add(KL.BatchNormalization(name='bn_conv1'))
    model.add(KL.Activation('relu'))
    model.add(KL.Flatten())
    model.add(KL.Dense(10))
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

