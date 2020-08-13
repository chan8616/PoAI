from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--n_clusters", type=int,
        metavar='Number of clusters',
        default=3,
        help='The number of clusters to form as well as the number of centroids to generate.',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    parser.add_argument(
        "--init", type=str,
        choices=['k-means++', 'random'],
        metavar='Init',
        default= 'k-means++',
        help='Method for initialization'
    )

    parser.add_argument(
        "--n_init", type=int,
        metavar='Number of init',
        default=10,
        help='Number of time the k-means algorithm will be run with different centroid seeds.',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    return parser
