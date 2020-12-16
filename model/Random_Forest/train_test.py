from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
import os
import pandas as pd

time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]

def train(config, X, Y, save_dir):
    print("Model build")
    model = RandomForestClassifier(n_estimators = config.n_estimators,
                                   criterion=config.criterion,
                                   max_depth=config.max_depth,
                                   min_samples_split=config.min_samples_split
                                   )
    print("Model fit")
    model.fit(X,Y)

    file_name = 'rf_model_' + time_stamp + '.sav'
    pickle.dump(model, open(os.path.join(save_dir, file_name), 'wb'))

    if config.save_predict:
        test(config, X, Y, save_dir, model)

def test(config, X, Y, save_dir, model):

    Y_predict = model.predict(X)
    print("Accuracy is: ", accuracy_score(Y_predict, Y))

    if config.save_predict:
        df_predict = pd.DataFrame({"predict": Y_predict})
        df_predict.to_csv(save_dir+'dt_model_' + time_stamp + '_predict.csv')


