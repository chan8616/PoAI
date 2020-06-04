from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import time
import os
import pandas as pd

time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]

def train(config, X, Y, save_dir):
    print("Model build")
    model = DecisionTreeClassifier(criterion=config.criterion,
                                   splitter= config.splitter,
                                   max_depth=config.max_depth,
                                   min_samples_split=config.min_samples_split,
                                   max_features=config.max_features,
                                   random_state=config.random_state,
                                   )
    print("Model fit")
    model.fit(X,Y)

    file_name = 'dt_model_' + time_stamp + '.sav'
    pickle.dump(model, open(os.path.join(save_dir, file_name), 'wb'))

    if config.save_figure or config.save_predict:
        test(config, X, Y, save_dir, model)

def test(config, X, Y, save_dir, model):

    Y_predict = model.predict(X)

    if config.save_predict:
        df_predict = pd.DataFrame({"predict": Y_predict})
        df_predict.to_csv(save_dir+'dt_model_' + time_stamp + '_predict.csv')

    if config.save_figure:
        import graphviz

        dot_data = tree.export_graphviz(model, out_file=None,
                                        feature_names = X.columns,
                                        filled = True, rounded = True,
                                        special_characters = True)

        graph = graphviz.Source(dot_data)
        file_name = 'dt_model_' + time_stamp
        graph.render(file_name, save_dir)
        os.remove(os.path.join(save_dir, file_name))


