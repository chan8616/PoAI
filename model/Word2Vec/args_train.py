from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--workers", type=int,
        metavar='The number of processors to be used for training',
        default=2
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
        default='word2vec.bin'
    )

    return parser


