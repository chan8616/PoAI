from argparse import Namespace
from gooey import GooeyParser


WEIGHTS = ['last']
LOSSES = ('binary_crossentropy '
          'categorical_crossentropy sparse_categorical_crossentropy '
          ).split()
OPTIMIZERS = ['sgd', 'adam']


class LogisticTrainConfig():
    WEIGHT = None
    FREEZE_LAYER = 0

    EPOCHS = 10

    LOSS = LOSSES[2]

    OPTIMIZER = OPTIMIZERS[0]
    LEARNING_RATE = .1
    LEARNING_MOMENTUM = .9

    MONITOR = 'loss'

    def update(self, train_args: Namespace):
        self.FREEZE_LAYER = train_args.freeze_layer
        self.EPOCHS = train_args.epochs

        self.LOSS = train_args.loss

        self.OPTIMIZER = train_args.optimizer
        self.LEARNING_RATE = train_args.learning_rate
        #  LEARNING_MOMENTUM = train_args.

def train_config_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        train_config=LogisticTrainConfig(),
        #  modifiable: bool = True,
        ) -> GooeyParser:

    load_parser = parser.add_mutually_exclusive_group()
    load_parser.add_argument(
        '--load_pretrained_weights',
        choices=WEIGHTS,
        #  default=train_config.WEIGHT,
        )
    #  load_parser.add_argument(
    #      '--load_specific_weights',
    #      choices=
    #      )
    load_parser.add_argument(
        '--load_pretrained_file',
        widget='FileChooser'
    )

    layers_parser = parser.add_argument_group()
    layers_parser.add_argument(
            '--freeze-layer',
            default=0,
            help="Number of freeze layers from the input."
            )

    steps_parser = parser.add_argument_group(
        title="Train Steps",
        description="Train Steps Setting",
        gooey_options={'columns': 3})

    steps_parser.add_argument(
        "epochs", type=int,
        default=train_config.EPOCHS,
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
    #          default=train_config.IMAGES_PER_GPU,
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
        "--loss",
        metavar='Loss',
        choices=LOSSES,
        default=train_config.LOSS,
    )
    compile_parser.add_argument(
            '--optimizer',
            choices=OPTIMIZERS,
            metavar='Optimizer',
            default=train_config.OPTIMIZER,
            )
    compile_parser.add_argument(
            '--learning_rate', type=eval,
            metavar='Learning rate',
            default=train_config.LEARNING_RATE,
            )

    return parser
