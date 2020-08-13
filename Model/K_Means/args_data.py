from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        'data_path',
        widget='FileChooser',
        metavar='Data Path',
        help="Data path (file path of .csv format)",
    )

    return parser
