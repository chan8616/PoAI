from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--input_length", type=int,
        metavar='Pads input sequences to the same length',
        default=30,
        help='if not provided, default value is the length of the longest sequence',
    )
    parser.add_argument(
        "--output_dim", type=int,
        metavar='Length of a word embedded vector',
        default=100,
        help='Define the size of the output vectors for each word',
    )
    parser.add_argument(
        "--n_lstm", type=int,
        metavar='The number of LSTM layers',
        default=1,
        help='Default: 1 LSTM layer',
    )
    parser.add_argument(
        "--n_class", type=int,
        metavar='The number of classes',
        default=2
    )
    parser.add_argument(
        "--early_stop",
        metavar='Early stopping',
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--metric", type=str,
        metavar='Quantity to monitor',
        default='val_loss',
        choices=['val_accuracy', 'val_loss', 'accuracy', 'loss']
    )
    parser.add_argument(
        "--loss", type=str,
        metavar='loss function',
        default='binary_crossentropy',
        choices=['binary_crossentropy', 'categorical_crossentropy']
    )
    parser.add_argument(
        "--epoch", type=int,
        metavar='Epochs',
        default=10
    )
    parser.add_argument(
        "--batch_size", type=int,
        metavar='Batch size',
        default=128,
        help='an arbitrary cutoff, generally defined as "one pass over the entire dataset"'
    )
    parser.add_argument(
        "--optimizer",
        metavar='Optimizer',
        default='RMSprop',
        choices=['RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adam']
    )
    parser.add_argument(
        "--lr", type=float,
        metavar='Learning rate',
        default=0.001,
    )
    parser.add_argument(
        "--patience", type=int,
        metavar="Patience",
        default=3,
        help="training will be stopped after the number of epochs with no improvement"
    )
    parser.add_argument(
        "--save_directory",
        metavar="Directory for saving checkpoints",
        widget="DirChooser",
        help="Checkpoint will be saved under 'this' directory"
    )
    parser.add_argument(
        "--ckpt_name",
        metavar="Checkpoint name",
        type=str,
        default='best_model.h5',
        help='checkpoint format must be .h5'
    )
    parser.add_argument(
        "--best_only",
        metavar='Save best model only',
        action="store_true",
        default=True
    )

    return parser


