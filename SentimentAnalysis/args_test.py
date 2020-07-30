from gooey import GooeyParser
from datetime import datetime


def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:
    parser.add_argument(
        "--checkpoint_path",
        metavar="Checkpoint",
        widget="FileChooser",
        help='Choose a checkpoint to test'
    )
    parser.add_argument(
        "--save_directory",
        metavar="Save directory",
        widget="DirChooser",
        help='Choose a directory to save the result '
    )
    parser.add_argument(
        "--test_fname",
        metavar="File name of the result",
        type=str,
        default='test_' + datetime.now().strftime('%Y%b%d_%H%M%S')+'.csv'
    )

    return parser
