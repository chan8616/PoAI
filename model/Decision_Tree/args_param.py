from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()):

    parser.add_argument(
        "--criterion", type=str,
        choices=['gini', 'entropy'],
        metavar='Criterion',
        default= 'gini',
        help='The function to measure the quality of a split.',
    )

    parser.add_argument(
        "--splitter", type=str,
        choices=['best', 'random'],
        metavar='Splitter',
        default= 'best',
        help='The strategy used to choose the split at each node.',
    )

    parser.add_argument(
        "--max_depth", type=int,
        metavar='Max depth',
        default=7,
        help='The maximum depth of the tree.',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    parser.add_argument(
        "--min_samples_split", type=int,
        metavar='Min samples split',
        default=2,
        help='The minimum number of samples required to split an internal node',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    parser.add_argument(
        "--max_features", type=int,
        metavar='Max features',
        default=None,
        help='The number of features to consider when looking for the best split',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    parser.add_argument(
        "--random_state", type=int,
        metavar='Random State',
        default=None,
        help='Controls the randomness of the estimator',
    )
