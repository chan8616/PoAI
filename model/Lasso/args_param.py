from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:
    parser.add_argument(
        "--alpha", type=float,
        metavar='to control the amount of shrinkage',
        default=0.01,
        help='The larger the value of alpha, the greater the amount of shrinkage',
        gooey_options={
            'validator': {
                'test': '0 <= float(user_input) <= 1',
                'message': 'Must be between 0 and 1.'
            }
        }
    )

    return parser
