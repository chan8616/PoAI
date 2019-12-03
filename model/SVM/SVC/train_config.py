from argparse import Namespace
from gooey import GooeyParser


class SVCTrainConfig():
    #  NAME = 'train'
    TRAIN_NAME = 'train'
    WEIGHT_PATH = None

    def update(self, args: Namespace):
        WEIGHT_PATH = args.load_pretrained_weights

    def _parser(self, parser=GooeyParser(),
                ) -> GooeyParser:
        title="Train Setting"

        load_parser = parser.add_mutually_exclusive_group()
        #  load_parser.add_argument(
        #      '--load_pretrained_weights',
        #      choices=WEIGHTS,
        #      #  default=self.WEIGHT,
        #      )
        #  load_parser.add_argument(
        #      '--load_specific_weights',
        #      choices=
        #      )
        load_parser.add_argument(
            '--load_pretrained_weights',
            widget = 'FileChooser'
            )

        return parser


class SVCTrain():
    def __init__(self, svc_train_config=SVCTrainConfig()):
        self.config = svc_train_config

    def train(self, model, train_generator, valid_generator, stream):
        """Train the model."""
        stream.put(('Loading...', None, None))
        #  if self.config.WEIGHT in WEIGHTS:
            #  weights_path = self.config.WEIGHT_PATH
        if self.config.WEIGHT_PATH:
            model = pickle.load(open(self.config.WEIGHT_PATH, 'rb'))
        #  model.load_weights(weights_path, by_name=True)
        train_df = train_generator
        stream.put(('Training', None, None))
        X, Y = train_df
        X = X.values
        y = Y.values.reshape(-1)
        model.fit(X, y)
        train_score = model.score(X, y)
        if valid_generator is not None:
            X, Y = valid_generator
            valid_score = model.score(X, y)
        else:
            valid_score = None
        stream.put(('end', None, None))
        return model
