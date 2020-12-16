import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
import time

def test(config):
    data_path = config.data_path
    save_directory = config.save_directory
    save_figure = config.save_figure
    pretrained_file_path = config.pretrained_file_path
    data = pd.read_csv(data_path)
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values.reshape(-1)
    X_test = X
    Y_test = Y

    pca = PCA(n_components=2)
    X_r = pca.fit(X_test).transform(X_test)

    model = pickle.load(open(pretrained_file_path, 'rb'))
    print("load pretrained model")


    if save_figure is True:

        Y_predict = model.predict(X_r)

        pred = pd.DataFrame(Y_predict)
        pred.columns = ['pred']
        feature = pd.DataFrame(X_r)
        p = pd.concat([feature, pred], axis=1)
        p.columns = ['f1', 'f2', 'pred']

        # centers = pd.DataFrame(model.cluster_centers_, columns=['f1', 'f2'])
        # center_x = centers['f1']
        # center_y = centers['f2']

        plt.figure(figsize=(10, 5))
        plt.scatter(p['f1'], p['f2'], c=p['pred'], alpha=0.5)
        # plt.scatter(center_x, center_y, s=50, marker='D', c='r')
        plt.title('Clustering result')

        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
        file_name = 'km_model_' + time_stamp + '.png'
        plt.savefig(os.path.join(save_directory, file_name))

