from gooey import GooeyParser
from datetime import datetime


def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:
    parser.add_argument(
        "--file_path",
        metavar="File directory",
        widget="DirChooser",
        help='Choose a directory which containing checkpoint file, vocab.json, and merges.txt'
    )

    parser.add_argument(
        "--input_sentence",
        metavar="Input sentence",
        type=str
    )

    return parser
