from sklearn.ensemble import RandomForestClassifier as rfc
from .model import NET
from os import path, makedirs
from copy import deepcopy

import sys
sys.path.append(path.abspath('..')) # refer to sibling folder

from .ops import *
from utils.util import pickle_save, pickle_load

class RF(NET):

    def __init__(self,
                dataset_name,
                checkpoint_dir,
                name,
                **kargs):
        model = 'randomforest'
        checkpoint_name = 'model'
        self.model_name = name
        checkpoint_dir = path.join(checkpoint_dir,model,dataset_name)
        if not path.exists(checkpoint_dir):
            makedirs(checkpoint_dir)
        self.model_dir = path.join(checkpoint_dir, self.model_name)
        self.model_ckpt = path.join(self.model_dir, checkpoint_name)
        self.model_meta = path.join(self.model_dir, 'meta')
        self.model_result = path.join('result', model)

        model_check = self.model_check()

        if model_check:
            self.model, self.model_conf = self.restore()
        else:
            self.model_conf = {'name':self.model_name,
                               'model_dir':self.model_dir,
                               'ckpt_path':self.model_ckpt,
                               'meta':self.model_meta,
                               'dataset':dataset_name,
                               'trained':False}
            self.build_model(self.model_conf)
            pickle_save(self.prog_info, self.model_meta)

    def restore(self):
        model = pickle_load(self.model_ckpt)
        conf = pickle_load(self.model_meta)
        return model, conf

    def save(self):
        pickle_save(self.model, self.model_ckpt)

    def build_model(self, conf):
        self.model = rfc(n_estimators=100, oob_score=True)

    def train(self,
              x,
              y,
              save=True, **kargs):
        print("[@] start training....")
        self.model.fit(x, y)
        self.model_conf['trained']= True
        if save:
            self.save()
        print("[@] Training is done...")

    def train_with_provider(self, generator, epochs, save=True):
        pass


    @property
    def prog_info(self):
        return deepcopy(self.model_conf)

    @property
    def trained(self):
        return self.model_conf['trained']
