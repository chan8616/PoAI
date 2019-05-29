import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int)

# Hyper-parameters
parser.add_argument("--input-dim", type=int, default=4) 
parser.add_argument("--output-dim", type=int, default=10)
parser.add_argument("--kernel", type=str, default='rbf')
parser.add_argument("--degree", type=int, default=3)
parser.add_argument("--gamma", type=str, default='auto')

# Paths
parser.add_argument("--save-path", type=str, )
parser.add_argument("--load-path", type=str, default=None,
                help='model load location, .npy')

# Others
parser.add_argument("--random_state")

if __name__ == '__main__':
    from module import MODULE
else:
    from .module import MODULE

class SVC(MODULE):
    def __init__(self):
        super(SVC, self).__init__('svc', None, 'Classification')
    
    def build(self, kernel, degree, gamma):
        from sklearn.svm import SVC
        self.model = SVC(kernel=kernel, degree=degree, gamma=gamma) 

    def save(self, path):
        pass
    def load(self, path):
        pass

    def fit(self, X, y):
        self.model.fit(X, y)

def main(parse):
    args, unknowns = parser.parse_known_args()
    print(args)

    model = SVC()
    if args.load_path is not None:
        model.load(args.load_path)
    else:
        model.build(args.kernel, args.degree, args.gamma)
    if args.mode == 'train':
        #        model.fit()
        if args.Xy is not None: 
            model.fit(*Xy)

    elif args.mode == 'test':

        model.eval()
        model.pred()


if __name__ == '__main__':
    main(parser)
