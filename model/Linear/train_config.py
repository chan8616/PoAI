from argparse import Namespace
from gooey import GooeyParser

from model.utils.stream_callbacks import KerasQueueLogger


WEIGHTS = ['last']
LOSSES = ['mean_squared_error']
OPTIMIZERS = ['sgd', 'adam']


class LinearTrainConfig():
    #  NAME = 'train'
    TRAIN_NAME = 'train'

    WEIGHT = None
    #  WEIGHT_PATH = None
    FREEZE_LAYER = 0

    EPOCHS = 10

    LOSS = LOSSES[0]

    OPTIMIZER = OPTIMIZERS[0]
    LEARNING_RATE = .1
    LEARNING_MOMENTUM = .9

    MONITOR = 'loss'

    def update(self, args: Namespace):
        self.WEIGHT = args.load_pretrained_weights
        #  self.WEIGHT_PATH = args.load_specific_weights
        self.FREEZE_LAYER = args.freeze_layer
        self.EPOCHS = args.epochs

        self.LOSS = args.loss

        self.OPTIMIZER = args.optimizer
        self.LEARNING_RATE = args.learning_rate
        #  LEARNING_MOMENTUM = args.

    def _parser(self, parser=GooeyParser(),
                ) -> GooeyParser:
        title="Train Setting"

        load_parser = parser.add_mutually_exclusive_group()
        load_parser.add_argument(
            '--load_pretrained_weights',
            choices=WEIGHTS,
            #  default=self.WEIGHT,
            )
        #  load_parser.add_argument(
        #      '--load_specific_weights',
        #      choices=
        #      )
        #  load_parser.add_argument(
        #      '--load_pretrained_weights',
        #      widget = 'FildChooser'
        #      )

        layers_parser = parser.add_argument_group()
        layers_parser.add_argument(
                '--freeze-layer', type=int,
                default=self.FREEZE_LAYER,
                help="Number of freeze layers from the input."
                )

        steps_parser = parser.add_argument_group(
            title="Train Steps",
            description="Train Steps Setting",
            gooey_options={'columns': 3})

        steps_parser.add_argument(
            "epochs", type=int,
            default=self.EPOCHS,
            help="number of training per entire dataset"
        )

        #  gpu_parser = parser.add_argument_group(
        #      title='GPU',
        #      description='GPU Setting',
        #      gooey_options={'columns': 3})

        #  gpu_parser.add_argument(
        #          '--gpu_list',
        #          # nargs="*",
        #          choices=get_available_gpus(),
        #          #  default=get_available_gpus(),
        #          metavar='GPU list',
        #          help="Avaiable GPU list.",
        #          #  widget="Listbox",
        #          )

        #  gpu_parser.add_argument(
        #          '--images_per_gpu', type=int,
        #          default=self.IMAGES_PER_GPU,
        #          metavar='Images per gpu',
        #          help="Number of images to train with on each GPU.\n"
        #               "A 12GB GPU can typically handle 2 images of 1024x1024px."
        #               "Adjust based on your GPU memory and image sizes.\n"
        #               "Use the highest number that your GPU can handle "
        #               "for best performance.",
        #          )

        compile_parser = parser.add_argument_group(
                title="Compile Settings",
                gooey_options={'show_border': True, 'columns': 2})
        compile_parser.add_argument(
            "--loss", type=str,
            metavar='Loss',
            choices=LOSSES,
            default=self.LOSS,
        )
        compile_parser.add_argument(
                '--optimizer', type=str,
                choices=OPTIMIZERS,
                metavar='Optimizer',
                default=self.OPTIMIZER,
                )
        compile_parser.add_argument(
                '--learning_rate', type=float,
                metavar='Learning rate',
                default=self.LEARNING_RATE,
                )

        return parser


class LinearTrain():
    def __init__(self, linear_train_config=LinearTrainConfig()):
        self.config = linear_train_config

    def train(self, model, train_generator, valid_generator, stream):
        """Train the model."""
        stream.put(('Loading...', None, None))
        if self.config.WEIGHT in WEIGHTS:
            if self.config.WEIGHT == "last":
                # find last trained weights
                weights_path = model.find_last()
            #  else:
            #      weights_path = self.config.WEIGHT_PATH

            model.load_weights(weights_path, by_name=True)

        stream.put(('Training', None, None))
        callbacks = [KerasQueueLogger(stream)]
        model.train(self.config,
                    train_generator, valid_generator,
                    custom_callbacks=callbacks)
