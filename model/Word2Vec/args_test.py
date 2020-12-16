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
        "--mode",
        metavar='Test mode',
        type=str,
        default='most_similar',
        choices=['most_similar', 'similarity', 'print'],
    )
    parser.add_argument(
        "--input_word",
        metavar="Test word",
        type=str,
        help='Choose a directory to save the result '
    )

    return parser
