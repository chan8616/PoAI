from konlpy import tag
import re


class Preprocessing:

    def __init__(self, config):
        self.data_path = config.data_path
        self.tagger_name = config.tagger_name
        self.size = config.size
        self.window = config.window
        self.min_count = config.min_count
        self.sg = config.sg
        self.workers = config.workers
        self.save_directory = config.save_directory
        self.ckpt_name = config.ckpt_name
        self.tagger = None
        self.commands = config.commands


    def tagger_load(self):
        name = self.tagger_name
        if name == 'Okt':
            self.tagger = tag.Okt()
        elif name == 'Kkma':
            self.tagger = tag.Kkma()
        elif name == 'Komoran':
            self.tagger = tag.Komoran()
        else:
            self.tagger = tag.Hannanum()
        print("load tagger")


    def tagging(self, data):
        pattern = '[^ㄱ-ㅎㅏ-ㅣ가-힣 ]'
        replace = ''
        sentences = []
        for sentence in data:
            clean = re.sub(pattern, replace, sentence)
            tagged = self.tagger.pos(clean, stem=True, norm=True)  # 토큰화
            temp = []
            if len(tagged) <= 0:
                continue
            for word in tagged:
                if word[1] in ['Noun', 'Verb']:
                    temp.append(word[0])
            sentences.append(temp)
        return sentences


    def do(self, data):
        # tagger 로드
        self.tagger_load()

        # 불용어 제거 및 토큰화
        print("Remove stopwords and tokenize input..")
        print("It may take time...")
        tokenized_data = self.tagging(data)

        # 리뷰 길이 분포 확인
        print('리뷰의 최대 길이 :', max(len(l) for l in tokenized_data))
        print('리뷰의 평균 길이 :', sum(map(len, tokenized_data)) / len(tokenized_data))

        return tokenized_data
