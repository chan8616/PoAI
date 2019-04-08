from tensorflow.python.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from .model import NET

_MIN_SIZE_ = 32

class DENSENET121(NET):
    def __init__(self, **kargs):
        kargs['model'] = 'densenet121'
        kargs['init'] = ['imagenet', 'random']
        if 'input_shape' in kargs.keys():
            kargs['input_shape'][0] = kargs['input_shape'][0] if kargs['input_shape'][0] > _MIN_SIZE_ else _MIN_SIZE_
            kargs['input_shape'][1] = kargs['input_shape'][1] if kargs['input_shape'][1] > _MIN_SIZE_ else _MIN_SIZE_
        super(DENSENET121, self).__init__(**kargs)

    def build_model(self, conf):
        base_model = DenseNet121(weights=conf['init'],
                           include_top = False,
                           pooling='avg',
                           classes=self.num_classes)
        x = base_model.output
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=base_model.input, outputs=y_pred)

        # if conf['freeze'] and conf['init'] is not 'random':
        # for layer in self.model.layers:
        for layer in base_model.layers:
            layer.trainable = True
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

class DENSENET169(NET):
    def __init__(self, **kargs):
        kargs['model'] = 'densenet169'
        kargs['init'] = ['imagenet', 'random']
        if 'input_shape' in kargs.keys():
            kargs['input_shape'][0] = kargs['input_shape'][0] if kargs['input_shape'][0] > _MIN_SIZE_ else _MIN_SIZE_
            kargs['input_shape'][1] = kargs['input_shape'][1] if kargs['input_shape'][1] > _MIN_SIZE_ else _MIN_SIZE_
        super(DENSENET169, self).__init__(**kargs)

    def build_model(self, conf):
        base_model = DenseNet169(weights=conf['init'],
                           include_top = False,
                           pooling='avg',
                           classes=self.num_classes)
        x = base_model.output
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=base_model.input, outputs=y_pred)

        # if conf['freeze'] and conf['init'] is not 'random':
        # for layer in self.model.layers:
        for layer in base_model.layers:
            layer.trainable = True
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

class DENSENET201(NET):
    def __init__(self, **kargs):
        kargs['model'] = 'densenet201'
        kargs['init'] = ['imagenet', 'random']
        if 'input_shape' in kargs.keys():
            kargs['input_shape'][0] = kargs['input_shape'][0] if kargs['input_shape'][0] > _MIN_SIZE_ else _MIN_SIZE_
            kargs['input_shape'][1] = kargs['input_shape'][1] if kargs['input_shape'][1] > _MIN_SIZE_ else _MIN_SIZE_
        super(DENSENET201, self).__init__(**kargs)

    def build_model(self, conf):
        base_model = DenseNet201(weights=conf['init'],
                           include_top = False,
                           pooling='avg',
                           classes=self.num_classes)
        x = base_model.output
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=base_model.input, outputs=y_pred)

        # if conf['freeze'] and conf['init'] is not 'random':
        # for layer in self.model.layers:
        for layer in base_model.layers:
            layer.trainable = True
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