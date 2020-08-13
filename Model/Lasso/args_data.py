from gooey import GooeyParser

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
        default = "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT",
        help='At least one columns must be selected.',
    )

    parser.add_argument(
        "--y_column", type=str,
        metavar='The y columns you want to use',
        default='MEDV',
        help='Only one column must be selected',
    )
    return parser
