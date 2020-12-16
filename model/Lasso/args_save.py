from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument( "--save-directory",
                         metavar="Save directory",
                         widget="DirChooser",
                         )
    return parser
