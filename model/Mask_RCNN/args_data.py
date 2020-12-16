from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--data_type",
        metavar="Choose data type",
        type=str,
        choices=['PennFudanPed', 'UserDataset'],
        default='UserDataset'
    )

    parser.add_argument(
        'root',
        widget='DirChooser',
        metavar='Data Path(root)',
        default=r'C:\Users\chanyang\Desktop\gui\TensorflowGUI-master\dataset\coil'
    )

    return parser
