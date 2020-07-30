from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        'data_path',
        widget='FileChooser',
        metavar='Data Path',
        help="Data path (file path of .txt format)"
    )

    parser.add_argument(
        "--size", type=int,
        metavar='The length of a embedded vector',
        default=100
    )

    parser.add_argument(
        #Okt(previous Twitter)
        "--tagger_name", type=str,
        choices=['Kkma', 'Komoran', 'Hannaunm', 'Okt'],
        metavar='Tagger',
        default='Okt',
        help='Select a tagger for POS tagging'
    )

    parser.add_argument(
        "--window", type=int,
        metavar='Context window size',
        default=5,
        help='Number of surrounding words to be considered when embedding words',
    )

    parser.add_argument(
        "--min_count", type=int,
        metavar='threshold of word frequency',
        default=3,
        help='Minimum occurrence of a word to be included in the vocabulary'
    )

    parser.add_argument(
        "--sg", type=str,
        choices=['CBOW', 'Skip-gram'],
        metavar='Strategy',
        default='Skip-gram',
        help='Select word embedding strategy '
    )

    return parser
