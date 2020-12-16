from build import *
from preprocessing import Preprocessing
from datasets import Datasets
import os


def do(config):
    # 데이터 읽기 & 전처리
    print("Read data")
    ds = Datasets(config.data_path)
    data = ds.read_data()

    print("Data preprocessing..")
    preprocessing = Preprocessing(config)
    x_train, y_train = preprocessing.do(data)

    print("Model build..")
    model, callback = build(config, preprocessing.vocab_size)

    history = model.fit(x_train, y_train, epochs=config.epoch, callbacks=callback, batch_size=config.batch_size, validation_split=0.2)
    model.save(os.path.join(config.save_directory, config.ckpt_name))