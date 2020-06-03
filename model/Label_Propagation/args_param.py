from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--kernel", type=str,
        choices=['knn', 'rbf'],
        metavar='Kernel',
        default= 'knn',
        help='Kernel Function',
    )

    parser.add_argument(
        "--gamma", type=float,
        metavar='Gamma',
        default=20.,
        help='parameter for rbf kernel',
    )

    parser.add_argument(
        "--n_neighbors", type=int,
        metavar='Number of neighbors',
        default=7,
        help='parameter for knn kernel',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    parser.add_argument(
        "--alpha", type=float,
        metavar='Alpha',
        default=0.2,
        help='Clamping factor',
        gooey_options={
            'validator': {
                'test': '0 <= float(user_input) <= 1',
                'message': 'Must be between 0 and 1.'
            }
        }
    )

    parser.add_argument(
        "--max_iter", type=int,
        metavar="Maximum Iteration",
        default=30,
        help="Maximum number of iterations allowed",
    )

    return parser
