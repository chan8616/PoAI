from konlpy import tag
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

class Preprocessing:

    def __init__(self, config):
        self.rm_duplicate = config.rm_duplicate
        self.tagger_name = config.tagger_name
        self.threshold = config.threshold
        self.max_sentence_len = config.max_sentence_len
        self.vocab_size = 0
        self.tokenizer = None
        self.save_directory = config.save_directory
        self.tokenizer_fname = config.tokenizer_fname
        self.tokenizer_path = config.tokenizer_path
        self.tagger = None
        self.commands = config.commands

    def save_tokenizer(self):
        with open(os.path.join(self.save_directory, self.tokenizer_fname), 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self):
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

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

    def remove_noise(self, data):
        # 중복 제거 및 중복 제거 후 데이터수 확인
        if self.rm_duplicate:
            n_duplicate_rows = len(data) - data['document'].nunique()
            print("# of duplicate rows: ", n_duplicate_rows)
            data['document'].drop_duplicates(inplace=True)
            print('# of remains:', len(data))

        # 한글과 공백을 제외하고 모두 제거
        print("Remove special characters(only Korean and white space will remain)")
        data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

        # Null 값이 존재하는 행 제거
        data['document'] = data['document'].str.strip()
        data['document'].replace('', np.nan, inplace=True)
        print(data.isnull().sum(), "empty rows exist..")
        data = data.dropna(how='any')
        data.reset_index(inplace=True)

        print("[Data Statistics]")
        print(data.groupby('label').size().reset_index(name='count'))
        return data

    def tagging(self, data):
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와']
        sentences = []
        for sentence in data['document']:
            temp = []
            temp = self.tagger.morphs(sentence, stem=True)  # 토큰화
            temp = [word for word in temp if not word in stopwords]  # 불용어 제거
            sentences.append(temp)

        return sentences

    def fit_tokenizer(self, X):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X)

        print("start counting frequency..")
        total_cnt = len(self.tokenizer.word_index)  # 단어의 수
        rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

        # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
        for key, value in self.tokenizer.word_counts.items():
            total_freq = total_freq + value

            # 단어의 등장 빈도수가 threshold보다 작으면
            if (value < self.threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        self.vocab_size = total_cnt - rare_cnt + 1

        print('the total size of vocabulary:', total_cnt)
        print('# of rare words(frequency of occurrence below the threshold): %d' % rare_cnt)
        print("% of rare words among all words:", (rare_cnt / total_cnt) * 100)
        print("the size of vocabulary without rare words: ", self.vocab_size)

        self.tokenizer = Tokenizer(self.vocab_size)
        self.tokenizer.fit_on_texts(X)

        self.save_tokenizer()

        return


    def int_encoding(self, X, y):
        X = self.tokenizer.texts_to_sequences(X)
        y = np.array(y)
        print(X[:3])

        # print("# of sentences: ", len(sentences), "# of data rows: ", len(data), "# of X(after tokenizer.texts_to_sequences): ", len(X))

        # 빈도수 낮은 단어 제거 후 문장 길이가 0인 데이터의 인덱스 확인
        drop_idx = [i for i, sentence in enumerate(X) if len(sentence) < 1]
        print('indices of (length 0)', drop_idx)
        # 학습데이터셋에서 빈 데이터 제거

        X = np.delete(X, drop_idx, axis=0)
        y = np.delete(y, drop_idx, axis=0)

        print(len(X), len(y))
        print('Max length of input: ', max(len(l) for l in X))
        print('Average length of input:', sum(map(len, X)) / len(X))

        return X, y

    def below_threshold_len(self, nested_list):
        cnt = 0
        for s in nested_list:
            if len(s) <= self.max_sentence_len:
                cnt = cnt + 1
        print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (self.max_sentence_len, (cnt / len(nested_list)) * 100))

    def do(self, data):
        # 노이즈 제거
        print("Remove noise from source data")
        clean_data = self.remove_noise(data)

        # tagger 로드
        self.tagger_load()

        # 불용어 제거 및 토큰화
        print("Remove stopwords and tokenize input..")
        X = self.tagging(clean_data)
        y = clean_data['label']

        # Vocabulary 생성
        if self.commands == 'train':
            self.fit_tokenizer(X)
        elif self.commands == 'test':
            self.load_tokenizer()
        X, y = self.int_encoding(X, y)

        # add padding to make the same size input
        self.below_threshold_len(X)
        X = pad_sequences(X, maxlen=self.max_sentence_len)
        y = to_categorical(y)

        return X, y
