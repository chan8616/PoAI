import tensorflow as tf
import sklearn
from model import simple

class Run():
    def __init__(self, spec):
        print(spec)
        # phase, model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
        # phase, model_spec, data_spec, train_spec
        # model_spec: name, path, 
        # data_spec: name, path, 
        # train_spec: checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed

        # train_spec: model, data, checkpoint, max_iter, batch_size, optimizer, lr, interval, random_seed
        # test_spec: model, data, # of images, max_per_number
        self.args = {}
        self.args['train'] = spec[0] == 'train'
        self.args['model'] = spec[1]
        if self.args['train']:
            self.args['dataset'] = spec[2]
            self.args['checkpoint'] = spec[3]
            self.args['max_iter'] = int(spec[4])
            self.args['batch_size'] = int(spec[1])
            self.args['optimizer'] = spec[1]
            self.args['lr'] = int(spec[1])
            self.args['interval'] = int(spec[1])
            self.args['random_seed'] = int(spec[1])
    
            if self.args['model']== 'logistic':
                self.logistic()
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
                    classes,
                    step_interval,
                    image_size,
                    train,
                    test,
                    dataset = args['dataset'],
                    checkpoint = args['checkpoint'],
                    epochs = args['max_iter']
                   )
