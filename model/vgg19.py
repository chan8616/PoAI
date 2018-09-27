from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D

from .model import NET

class VGGNET19(NET):

    def __init__(self, **kargs):
        kargs['model'] = 'vgg19'
        kargs['init'] = ['imagenet', 'random']
        super(VGGNET19, self).__init__(**kargs)

    def build_model(self, conf):
        base_model = VGG19(weights=conf['init'], include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.add_layer[0], activation='relu', name='lin1')(x)
        y_pred = Dense(self.num_classes, activation='softmax', name='prediction')(x)
        self.model = Model(inputs=base_model.input, outputs=y_pred)

        if conf['freeze'] and conf['init'] is not 'random':
            for layer in base_model.layers:
                layer.trainable = False
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
