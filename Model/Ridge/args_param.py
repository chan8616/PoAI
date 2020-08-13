from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:
    parser.add_argument(
        "--alpha", type=float,
        metavar='to control the amount of shrinkage',
        default=0.01,
        help='The larger the value of alpha, the greater the amount of shrinkage',
        gooey_options={
            'validator': {
                'test': 'float(user_input) > 0',
                'message': 'Must be positive number.'
            }
        }
    )
    return parser
