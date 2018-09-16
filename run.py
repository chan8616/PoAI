import tensorflow as tf
import sklearn
from model import simple

class Run():
    def __init__(self, spec):
        # model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
        print(spec)
        if spec[0] == 'train':
            pass
        elif spec[0] == 'test':
            pass
        else:
            print('spec error')
        pass
    
    def svm(self):
       pass 

    def logistic(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            simple.load_model(sess,
                    dataset,
                    epochs,
                    classes,
                    step_interval,
                    image_size,
                    train,
                    test
                    )
