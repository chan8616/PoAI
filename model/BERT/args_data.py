from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        'files',
        widget='FileChooser',
        metavar='Data Path',
        help="Corpus Data(.txt format file)"
    )

    parser.add_argument(
        "--save_directory",
        metavar="Save directory",
        widget="DirChooser",
        help='Choose a directory to save the checkpoint and vocab files '
    )

    parser.add_argument(
        "--add_prefix_space",
        metavar="Add prefix space",
        action="store_true",
        default=True,
        help='prefix to separate words'
                        )

    parser.add_argument(
        "--trim_offsets",
        metavar='Trim offsets',
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--vocab_size", type=int,
        metavar='Vocabulary size',
        default=50000,
        help='the maximum number of words to be included in the vocabulary'
    )

    parser.add_argument(
        "--min_frequency", type=int,
        metavar='Minimum frequency',
        default=2,
        help='Minimum word occurrence to be included in the vocabulary'
    )

    parser.add_argument(
        "--limit_alphabet", type=int,
        metavar='Limit alphabet',
        default=6000
    )

    parser.add_argument(
        "--special_tokens", type=str,
        metavar="Special tokens",
        action='append',
        default=["<s>","<pad>","</s>","<unk>","<mask>"]
    )

    return parser
