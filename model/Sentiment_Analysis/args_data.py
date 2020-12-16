from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        'data_path',
        widget='FileChooser',
        metavar='Data Path',
        help="Data path (file path of .txt format)"
    )

    parser.add_argument(
        #Okt(previous Twitter)
        "--tagger_name", type=str,
        choices=['Kkma', 'Komoran', 'Hannaunm', 'Okt'],
        metavar='Tagger',
        default='Okt',
        help='Select a tagger for POS tagging'
    )

    parser.add_argument("--tokenizer_fname",
                        metavar="Tokenizer file name(Train mode only)",
                        type=str,
                        default='tokenizer.pkl',
                        help='tokenizer will be saved with this name'
                        )

    parser.add_argument(
        '--tokenizer_path',
        widget='FileChooser',
        metavar='Tokenizer path(Test mode only)',
        help="please select a tokenizer file to use.."
    )

    parser.add_argument(
        "--rm_duplicate",
        metavar='Remove duplicate data',
        action="store_true",
        default=True,
        help='Check if you want to remove duplicate rows'
    )

    parser.add_argument(
        "--threshold", type=int,
        metavar='threshold of word frequency',
        default=3,
        help='Minimum occurrence of a word to be included in the vocabulary'
    )

    parser.add_argument(
        "--max_sentence_len", type=int,
        metavar='max length of a sentence(should be exactly the same with model input size)',
        default=30
    )
    return parser
