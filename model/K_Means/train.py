import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import time
import os

def train(config):

    n_clusters = config.n_clusters
    init = config.init
    n_init = config.n_init

    data_path = config.data_path
    save_directory = config.save_directory
    save_figure = config.save_figure
    data = pd.read_csv(data_path)
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values.reshape(-1)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    X_train = X
    Y_train = Y

    print("Model build")

    pca = PCA(n_components=2)
    X_r = pca.fit(X_train).transform(X_train)

    model = cluster.KMeans(init=init, n_clusters=n_clusters, n_init=n_init)
    model.fit(X_r, Y)
    print("Model fit")

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'km_model_' + time_stamp + '.sav'
    pickle.dump(model, open(os.path.join(save_directory, file_name), 'wb'))


    if save_figure is True:

        Y_predict = model.predict(X_r)

        pred = pd.DataFrame(Y_predict)
        pred.columns = ['pred']
        feature = pd.DataFrame(X_r)
        p = pd.concat([feature, pred], axis=1)
        p.columns = ['f1', 'f2', 'pred']

        centers = pd.DataFrame(model.cluster_centers_, columns=['f1', 'f2'])
        center_x = centers['f1']
        center_y = centers['f2']

        plt.figure(figsize=(10, 5))
        plt.scatter(p['f1'], p['f2'], c=p['pred'], alpha=0.5)
        plt.scatter(center_x, center_y, s=50, marker='D', c='r')
        plt.title('Clustering result')

        file_name = 'km_model_' + time_stamp + '.png'
        plt.savefig(os.path.join(save_directory, file_name))