# -*- coding: utf-8 -*-

import os
from transformers import pipeline
from tokenizers import ByteLevelBPETokenizer

def do(config):
    print("user input sentence: ", config.input_sentence)
    input_sentence = config.input_sentence + ' ' + '<mask>'

    #model load
    model = pipeline(
        "fill-mask",
        model=config.file_path,
        tokenizer=config.file_path
    )

    #make prediction
    #preds = []
    next_words = model(input_sentence)
    for i, word in enumerate(next_words):
        '''
        temp = word['sequence'][3:-4]
        preds.append(temp.split()[-1])
        '''
        print(("prediction %d: " % i), word['sequence'][3:-4])
