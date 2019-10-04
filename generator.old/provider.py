from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

def provider(path):
    if 'mnist' == name:
        (x_train, y_train), (x_test, y_test) = mnist.load_dataset()
        gen = ImageDataGenerator(rescale=1./255)
    return gen.flow()

def Xy(name):
    pass

def preprocessing(provider):
    pass

def imagepreprocessing(provider):
    pass
