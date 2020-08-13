from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import time
import os

def train(config):

    data_path = config.data_path
    save_directory = config.save_directory
    data = pd.read_csv(data_path)

    x_columns = config.x_columns
    x_columns = x_columns.split(',')
    X = data[x_columns]
    y_column = config.y_column
    Y = data[y_column]

    X_train = X
    Y_train = Y

    print("Model build")

    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    print("Model fit")

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'linear_model_' + time_stamp + '.sav'
    pickle.dump(lin_model, open(os.path.join(save_directory, file_name), 'wb'))

    y_train_predict = lin_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

