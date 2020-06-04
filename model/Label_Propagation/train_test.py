from sklearn.semi_supervised import LabelSpreading
import pickle
import time
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]

def train(config, X, Y, save_dir):
    print("Model build")
    model = LabelSpreading(alpha=config.alpha,
                           gamma= config.gamma,
                           kernel=config.kernel,
                           max_iter=config.max_iter,
                           n_neighbors=config.n_neighbors)
    print("Model fit")
    model.fit(X,Y)

    file_name = 'lp_model_' + time_stamp + '.sav'
    pickle.dump(model, open(os.path.join(save_dir, file_name), 'wb'))

    if config.save_figure or config.save_predict:
        test(config, X, Y, save_dir, model)

def test(config, X, Y, save_dir, model):

    Y_predict = model.predict(X)

    if config.save_predict:
        df_predict = pd.DataFrame({"predict": Y_predict})
        df_predict.to_csv(save_dir+'lp_model_' + time_stamp + '_predict.csv')

    if config.save_figure:
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(X)

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
        plt.savefig(os.path.join(save_dir, file_name))

