from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) :

    parser.add_argument(
        'data_path',
        widget='FileChooser',
        metavar='Data Path',
        help="Data path (file path of .csv format)",
        gooey_options={
            'validator': {
                'test': 'user_input.split(".")[-1] == "csv"',
                'message': 'Data format must be "csv".'
            }
        }
    )
