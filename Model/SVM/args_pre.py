from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument('--pretrained_file_path',
                        widget='FileChooser',
                        metavar='Pretrained model file Path',
                        help="Pretrained model file path (file path of .sav format)",
                        )

    return parser