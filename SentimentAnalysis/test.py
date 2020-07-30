from build import *
from preprocessing import Preprocessing
from datasets import Datasets
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

def do(config):
    # 데이터 읽기 & 전처리
    print("Read data")
    ds = Datasets(config.data_path)
    data = ds.read_data()

    print("Data preprocessing..")
    preprocessing = Preprocessing(config)
    x_test, y_test = preprocessing.do(data)

    print("Model build..")
    model = load_model(config.checkpoint_path)

    score = model.predict(x_test)
    pred = list(np.argmax(score, axis=1))
    original_sentence = preprocessing.tokenizer.sequences_to_texts(x_test)
    print(original_sentence)
    gt = list(np.argmax(y_test, axis=1))
    result = pd.DataFrame({"sentence":original_sentence, "prediction":pred, 'ground truth':gt})
    result.to_csv(os.path.join(config.save_directory, config.test_fname), encoding='utf-8-sig')
