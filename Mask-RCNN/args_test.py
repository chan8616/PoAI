from gooey import GooeyParser
from datetime import datetime


def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:

    parser.add_argument(
        "--checkpoint_path",
        metavar="Checkpoint",
        widget="FileChooser",
        help='Choose a checkpoint to test',
        default=r'C:\Users\chanyang\Desktop\gui\TensorflowGUI-master\Mask-RCNN\model.pth'
    )

    parser.add_argument(
        '--save_directory',
        widget='DirChooser',
        metavar='Save directory for test result',
        default=r'C:\Users\chanyang\Desktop\gui\TensorflowGUI-master\Mask-RCNN\Result'
    )

    return parser
