# -*- coding: utf-8 -*-

import os
from gensim.models import Word2Vec

def do(config):
    model = Word2Vec.load(config.checkpoint_path)
    print('model shape: ', model.wv.vectors.shape)

    if config.mode == 'most_similar':
        result = model.wv.most_similar(config.input_word)
    elif config.mode == 'similarity':
        words = config.input_word.split(' ')
        result = model.wv.similarity(words[0], words[1])
    else:
        result = model.wv[config.input_word]

    print(result)