import os
from tokenizers import ByteLevelBPETokenizer

class Preprocessing:

    def __init__(self, config):
        self.files = config.files
        self.save_directory = config.save_directory
        self.add_prefix_space = config.add_prefix_space
        self.trim_offsets = config.trim_offsets
        self.vocab_size = config.vocab_size
        self.min_frequency = config.min_frequency
        self.limit_alphabet = config.limit_alphabet
        self.special_tokens = config.special_tokens
        self.commands = config.commands

    def create_tokenizer(self):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=self.files,
                        vocab_size=self.vocab_size,
                        min_frequency=self.min_frequency,
                        special_tokens=self.special_tokens)
        vocab_path = os.path.join(self.save_directory)
        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)
        tokenizer.save_model(vocab_path)

        return tokenizer

    def do(self):
        # create vocab.txt, merges.txt files
        tokenizer = self.create_tokenizer()
        return
