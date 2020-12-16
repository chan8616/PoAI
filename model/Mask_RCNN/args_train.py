from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--batch_size", type=int,
        metavar='Batch size',
        default=2
    )

    parser.add_argument(
        "--num_epochs", type=int,
        metavar='Epochs',
        default=10
    )

    parser.add_argument(
        "--num_classes", type=int,
        metavar='Number of classes',
        default=2
    )

    parser.add_argument(
        "--lr", type=float,
        metavar='Learning rate',
        default=0.005
    )

    parser.add_argument(
        "--weight_decay", type=float,
        metavar='Weight decay',
        default=0.0005
    )

    parser.add_argument(
        "--momentum", type=float,
        metavar='Momentum',
        default=0.9
    )

    parser.add_argument(
        "--step_size", type=int,
        metavar='Stepsize',
        default=3
    )

    parser.add_argument(
        "--gamma", type=float,
        metavar='Gamma',
        default=0.1
    )

    parser.add_argument(
        "--save_directory",
        metavar="Directory for saving checkpoints",
        widget="DirChooser",
        help="Checkpoint will be saved under 'this' directory"
    )

    parser.add_argument(
        "--ckpt_name",
        metavar="Checkpoint file name(file format must be .bin)",
        type=str,
        default='model.pth'
    )

    return parser


