from argparse import Namespace
from gooey import GooeyParser
from pathlib import Path

from .config import Config

MODEL_DIR = Path("checkpoint/BERT/")


def build_config_parser(
        parser: GooeyParser = GooeyParser(description='Build Option'),
        config: Config = Config(),
        modifiable: bool = True,
        ) -> GooeyParser:

    require_parser = parser.add_argument_group('requires')

    require_parser.add_argument(
        '--bert_config_file',
        metavar='Config json file of pre-trained BERT model',
        help="Config json file of pre-trained BERT model",
        widget='FileChooser',
    )

    require_parser.add_argument(
        '--vocab_file',
        metavar='Vocab txt file of pre-trained BERT model',
        help="The vocabulary file that the BERT model was trained on",
        widget='FileChooser',
    )

    # require_parser.add_argument(
    #     '--output_dir',
    #     metavar='Output directory',
    #     help="The output directory where the model checkpoints will be written",
    #     widget='DirChooser',
    # )

    log_parser = parser.add_argument_group(
        'Log',
        "Show and Save model options",
        gooey_options={'show_border': True, 'columns': 4}
        )

    log_parser.add_argument(
        "--log-dir", type=str,
        metavar="Log Directory Path",
        default=(MODEL_DIR.joinpath('untitled') if config.NAME is None
                 else MODEL_DIR.joinpath(str(config.NAME))),
        help='{}{}TIME{}/'.format(
            MODEL_DIR.joinpath('LOG_NAME'),
            '{', '}'),
        widget='DirChooser'
        )

    #  show_and_save_parser.add_argument(
    #      "--save-path", type=str,
    #      metavar="File path",
    #      default="model.h5",
    #      help="model name to save model",
    #      )
    #  show_and_save_parser.add_argument(
    #      "--save-file", type=str,
    #      metavar="Overwrite File",
    #      help="model name to save model",
    #      widget="FileChooser",
    #      )

    return parser


def build_config(args: Namespace) -> Config:
    class BuildConfig(Config):
        NAME = Path(args.log_dir).name
        BERT_CONFIG_FILE = args.bert_config_file
        VOCAB_FILE = args.vocab_file
        OUTPUT_DIR = args.log_dir

    return BuildConfig()
