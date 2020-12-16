from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import time
import os

def test(config):
    data_path = config.data_path
    save_directory = config.save_directory
    pretrained_file_path = config.pretrained_file_path
    data = pd.read_csv(data_path)

    x_columns = config.x_columns
    x_columns = x_columns.split(',')
    X = data[x_columns]
    y_column = config.y_column
    Y = data[y_column]

    X_test = X
    Y_test = Y

    model = pickle.load(open(pretrained_file_path, 'rb'))
    print("load pretrained model")

    y_test_predict = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))

    dataframe = pd.DataFrame(y_test_predict)
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'ridge_predict_result' + time_stamp + '.csv'
    dataframe.to_csv(os.path.join(save_directory, file_name), header = False, index = False)

