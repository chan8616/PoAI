import os
from build import build
from preprocessing import Preprocessing

def do(config):
    # 데이터 읽기 & 전처리
    print("Make vocab.json and merges.txt")
    preprocessing = Preprocessing(config)
    preprocessing.do()

    print("Model build..")
    trainer = build(config)

    trainer.train()
    trainer.save_model(os.path.join(config.save_directory))

    # create empty modelcard file.. >> to fix transformers library bug
    f = open(file=os.path.join(config.save_directory, 'modelcard.json'), mode='w')
    f.close()

    print("Training complete!")