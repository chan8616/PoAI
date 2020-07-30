from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--max_length", type=int,
        metavar='maximum length of input',
        default=64,
        help='Pad or truncate input sequences to the same length',
    )

    parser.add_argument(
        "--num_attention_heads", type=int,
        metavar='the number of attention heads',
        default=12
    )

    parser.add_argument(
        "--num_hidden_layers", type=int,
        metavar='The number of hidden layers',
        default=6
    )

    parser.add_argument(
        "--mlm_probability", type=float,
        metavar='Ratio of <mask> in the whole corpus',
        default=0.15
    )

    parser.add_argument(
        "--num_train_epochs", type=int,
        metavar='Epochs',
        default=10
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", type=int,
        metavar='Batch size per GPU',
        default=32
    )

    parser.add_argument(
        "--save_steps", type=int,
        metavar='Save steps',
        default=10_000
    )

    parser.add_argument(
        "--save_total_limit", type=int,
        metavar='Save total limit',
        default='2'
    )

    parser.add_argument(
        "--prediction_loss_only",
        metavar='Prediction loss only',
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--overwrite_output_dir",
        metavar='Overwrite output directory',
        action="store_true",
        default=True
    )
    return parser


