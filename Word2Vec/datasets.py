class Datasets:
    def __init__(self, data_path):
        self.path = data_path

    def read_data(self):
        f = open(self.path, encoding="utf8", mode='r')
        data = f.readlines()
        f.close()
        return data