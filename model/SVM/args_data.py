from gooey import GooeyParser
import textwrap

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        'data_path',
        widget='FileChooser',
        metavar='Data Path',
        help="Data path (file path of .csv format)",
    )

    parser.add_argument(
        "--x_columns", type=str,
        metavar='The x columns you want to use',
        default = "sepal length (cm),sepal width (cm)",
        help= textwrap.dedent('''\
        At least one columns must be selected.
        (If you want to draw a plot, you have to choose just two variables.)'''),
    )

    parser.add_argument(
        "--y_column", type=str,
        metavar='The y columns you want to use',
        default='label',
        help='Only one column must be selected',
    )

    return parser
