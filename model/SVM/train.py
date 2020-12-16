from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import time
import os

def train(config):

    kernel = config.kernel
    C =config.C
    degree = config.degree

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

    svm_model = SVC(kernel=kernel, C=C, degree=degree)
    svm_model.fit(X_train, Y_train)

    print("Model fit")

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'svm_model_' + time_stamp + '.sav'
    pickle.dump(svm_model, open(os.path.join(save_directory, file_name), 'wb'))

    print("The model performance for training set")
    print("--------------------------------------")
    print("\n")

