import pandas as pd


class Datasets:
    def __init__(self, data_path):
        self.path = data_path

    def read_data(self):
        data = pd.read_table(self.path)
        if len(data.columns) == 2:
            data.columns=['document', 'label']
        elif len(data.columns) == 3:
            data.columns = ['id', 'document', 'label']
            data.set_index('id')
        else:
            print("Please check data shape..!")
            raise ValueError('Data shape does not fit!')
        return data
