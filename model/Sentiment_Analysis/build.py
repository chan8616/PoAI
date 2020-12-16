import os
from tensorflow.keras import losses
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 모델 build
def build(config, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, config.output_dim))
    for i in range(config.n_lstm):
        model.add(LSTM(128))
    model.add(Dense(config.n_class, activation='softmax'))

    # define loss function
    if config.loss == 'binary_crossentropy':
        loss = losses.BinaryCrossentropy()
    elif config.loss == 'categorical_crossentropy':
        loss = losses.CategoricalCrossentropy()

    model.compile(optimizer=config.optimizer, loss=loss)

    es = EarlyStopping(monitor=config.metric, mode='auto', verbose=1, patience=config.patience)
    mc = ModelCheckpoint(os.path.join(config.save_directory, config.ckpt_name), monitor=config.metric,
                         mode='auto', verbose=1, save_best_only=config.best_only)
    callback = [es, mc]
    model.summary()
    return model, callback