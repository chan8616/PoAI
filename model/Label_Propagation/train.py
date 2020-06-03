from sklearn.semi_supervised import LabelSpreading
import pandas as pd
import pickle
import time
import os


def train(config):

    alpha = config.alpha
    gamma = config.gamma
    kernel = config.kernel
    max_iter = config.max_iter
    n_neighbors = config.n_neighbors

    data_path = config.data_path
    save_directory = config.save_directory
    save_figure = config.save_figure
    data = pd.read_csv(data_path)
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values.reshape(-1)

    print("Model build")
    model = LabelSpreading(alpha=alpha,
                           gamma= gamma,
                           kernel=kernel,
                           max_iter=max_iter,
                           n_neighbors=n_neighbors)
    print("Model fit")
    model.fit(X,Y)

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'lp_model_' + time_stamp + '.sav'
    pickle.dump(model, open(os.path.join(save_directory, file_name), 'wb'))

    if save_figure is True:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import seaborn as sns

        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)
        Y_predict = model.predict(X)

        hue_order = sorted(set(Y), reverse=True)
        markers = {i: 's' for i in hue_order}
        markers[-1] = "X"
        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X_r[:, 0], y=X_r[:, 1],
                        hue=Y, style=Y, palette="Set2",
                        markers=markers, hue_order=hue_order,
                        legend=False)
        plt.title('Before some unlabeled')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X_r[:, 0], y=X_r[:, 1],
                        hue=Y_predict, style=Y_predict, palette="Set2",
                        markers=markers, hue_order=hue_order[:-1])
        plt.title('After Predict')

        plt.suptitle("Unlabeled points are marked 'X'")

        file_name = 'lp_model_' + time_stamp + '.png'
        plt.savefig(os.path.join(save_directory, file_name))



