from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()):

    parser.add_argument( "--save-figure",
                         metavar="Save result figure",
                         action='store_true',
                         )

    parser.add_argument( "--save-predict",
                         metavar="Save predict CSV",
                         action='store_true',
                         )
