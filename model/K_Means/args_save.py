from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument( "--save-directory",
                         metavar="Save directory",
                         widget="DirChooser",
                         )

    parser.add_argument( "--save-figure",
                         metavar="Save result figure",
                         action='store_true',
                         )
    return parser
