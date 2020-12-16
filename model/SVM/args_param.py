from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:
    parser.add_argument(
        "--kernel",
        metavar='kernel',
        help='Specifies the kernel type to be used in the algorithm.',
        choices=['linear','poly','rbf']
    )

    parser.add_argument(
        "--C", type=float,
        metavar='Regularization parameter',
        default=1.0,
        help='The strength of the regularization is inversely proportional to C.',
        gooey_options={
            'validator': {
                'test': 'float(user_input) > 0',
                'message': 'Must be positive number.'
            }
        }
    )

    parser.add_argument(
        "--degree", type=int,
        metavar='Degree',
        default=3,
        help='Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive number.'
            }
        }
    )

    return parser
