from sklearn import svm
from .model import NET

class SVM(NET):

    def __init__(self,
                dataset_name,
                checkpoint_dir,
                name,
                **kargs):
        model = 'svm'
        checkpoint_name = 'model'
        self.model_name = '{}_{}'.format(dataset_name, name) if name is not None else dataset_name
        checkpoint_dir = path.join(checkpoint_dir, model)
        if not path.exists(checkpoint_dir):
            makedirs(checkpoint_dir)
        self.model_dir = path.join(checkpoint_dir, self.model_name)
        self.model_ckpt = path.join(self.model_dir, checkpoint_name)
        self.model_meta = path.join(self.model_dir, 'meta')

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
            self.trained = False
            self.build_model(self.model_conf)
            pickle_save(self.prog_info, self.model_meta)

    def restore(self):
        model = pickle_load(self.model_ckpt)
        conf = pickle_load(self.model_meta)
        return model, conf

    def save(self):
        pickle_save(self.model, self.model_ckpt)

    def build_model(self, conf):
        self.model = svm.SVC()

    def train(self,
              x,
              y,
              save=True, **kargs):

        self.model.fit(x, y)
        self.trained = True
        if save:
            self.save()

    def train_with_provider(self, generator, epochs, save=True):
        pass

    @property
    def prog_info(self):
        return deepcopy(self.model_conf)
