from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from .model import NET

_MIN_SIZE_ = 32

class VGGNET19(NET):
    def __init__(self, **kargs):
        kargs['model'] = 'vgg19'
        kargs['init'] = ['imagenet', 'random']
        if 'input_shape' in kargs.keys():
            kargs['input_shape'][0] = kargs['input_shape'][0] if kargs['input_shape'][0] > _MIN_SIZE_ else _MIN_SIZE_
            kargs['input_shape'][1] = kargs['input_shape'][1] if kargs['input_shape'][1] > _MIN_SIZE_ else _MIN_SIZE_
        super(VGGNET19, self).__init__(**kargs)

    def build_model(self, conf):
        base_model = VGG19(weights=conf['init'],
                           include_top=False,
                           pooling='avg',
                           classes=self.num_classes)
        x = base_model.output
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=base_model.input, outputs=y_pred)
        # if conf['freeze'] and conf['init'] is not 'random':
        # for layer in self.model.layers:
        for layer in base_model.layers:
            layer.trainable = False

        if conf['optimizer'] == 'adam':
            # optimizer = keras.optimizers.Adam(lr=conf['learning_rate'])
            # self.model.compile(optimizer='adam', loss='categorical_crossentropy',
            self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        elif conf['optimizer'] == 'rmsprop':
            # optimizer = keras.optimizers.RMSprop(lr=conf['learning_rate'])
            # self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        elif conf['optimizer'] == 'adagrad':
            # optimizer = keras.optimizers.Adagrad(lr=conf['learning_rate'])
            self.model.compile(optimizer='adagrad', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        elif conf['optimizer'] == 'adadelta':
            # optimizer = keras.optimizers.Adadelta(lr=conf['learning_rate'])
            self.model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            # optimizer = keras.optimizers.SGD(lr=conf['learning_rate'])
            self.model.compile(optimizer='sgd', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        #self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


class VGGNET16(NET):
    def __init__(self, **kargs):
        kargs['model'] = 'vgg16'
        kargs['init'] = ['imagenet', 'random']
        if 'input_shape' in kargs.keys():
            kargs['input_shape'][0] = kargs['input_shape'][0] if kargs['input_shape'][0] > _MIN_SIZE_ else _MIN_SIZE_
            kargs['input_shape'][1] = kargs['input_shape'][1] if kargs['input_shape'][1] > _MIN_SIZE_ else _MIN_SIZE_
        super(VGGNET16, self).__init__(**kargs)

    def build_model(self, conf):
        base_model = VGG16(weights=conf['init'],
                           include_top = False,
                           pooling='avg',
                           classes=self.num_classes)
        x = base_model.output
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=base_model.input, outputs=y_pred)

        # if conf['freeze'] and conf['init'] is not 'random':
        # for layer in self.model.layers:
        for layer in base_model.layers:
            layer.trainable = False
        if conf['optimizer'] == 'adam':
            # optimizer = keras.optimizers.Adam(lr=conf['learning_rate'])
            # self.model.compile(optimizer='adam', loss='categorical_crossentropy',
            self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        elif conf['optimizer'] == 'rmsprop':
            # optimizer = keras.optimizers.RMSprop(lr=conf['learning_rate'])
            # self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        elif conf['optimizer'] == 'adagrad':
            # optimizer = keras.optimizers.Adagrad(lr=conf['learning_rate'])
            self.model.compile(optimizer='adagrad', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        elif conf['optimizer'] == 'adadelta':
            # optimizer = keras.optimizers.Adadelta(lr=conf['learning_rate'])
            self.model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            # optimizer = keras.optimizers.SGD(lr=conf['learning_rate'])
            self.model.compile(optimizer='sgd', loss='categorical_crossentropy',
                               metrics=['accuracy'])
        # self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])