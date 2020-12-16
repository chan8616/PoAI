import os
from preprocessing import Preprocessing
from datasets import Datasets
from gensim.models import Word2Vec


def do(config):
    # 데이터 읽기 & 전처리
    print("Read data")
    ds = Datasets(config.data_path)
    data = ds.read_data()

    print("Data preprocessing..")
    preprocessing = Preprocessing(config)
    X = preprocessing.do(data)

    print('Train model')

    if config.sg == 'CBOW':
        model = Word2Vec(
                    sentences=X,
                    size=config.size,
                    window=config.window,
                    min_count=config.min_count,
                    workers=config.workers,
                    sg=0
        )
    else:
        model = Word2Vec(
            sentences=X,
            size=config.size,
            window=config.window,
            min_count=config.min_count,
            workers=config.workers,
            sg=1
        )

    print(model.wv.vectors.shape)

    model.save(os.path.join(config.save_directory, config.ckpt_name))
