from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def test(config):
    data_path = config.data_path
    save_directory = config.save_directory
    save_figure = config.save_figure

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
    acc = accuracy_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('accuracy score is {}'.format(acc))

    if save_figure is True:
        X_test_a = np.array(X_test)

        h = .02  # step size in the mesh
        x_min, x_max = X_test_a[:, 0].min() - 1, X_test_a[:, 0].max() + 1
        y_min, y_max = X_test_a[:, 1].min() - 1, X_test_a[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X_test_a[:, 0], X_test_a[:, 1], c=Y_test, cmap=plt.cm.Paired, edgecolors='k')
        plt.title('classification result')
        plt.axis('tight')

        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
        file_name = 'svm_model_' + time_stamp + '.png'
        plt.savefig(os.path.join(save_directory, file_name))

