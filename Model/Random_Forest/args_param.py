from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()):

    parser.add_argument(
        "--n_estimators", type=int,
        metavar='n_estimators',
        default=10,
        help='The number of trees in the forest.',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )

    parser.add_argument(
        "--criterion", type=str,
        choices=['gini', 'entropy'],
        metavar='Criterion',
        default= 'gini',
        help='The function to measure the quality of a split.',
    )

    parser.add_argument(
        "--max_depth", type=int or str,
        metavar='Max depth',
        default=None,
        help='The maximum depth of the tree.',
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


