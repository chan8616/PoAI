# type: ignore
from pathlib import Path
from typing import Union
import pickle
from argparse import ArgumentParser, _ArgumentGroup
from gooey import Gooey, GooeyParser

from sklearn.svm import SVC as baseModel

DEFAULT_PARAMS = {'C': 1.0,
                  'kernel': 'rbf',
                  'degree': 3,
                  'gamma': 'auto',
                  'coef0': 0.0}
MODEL_DIR = "checkpoint/svm/"


def build_parser(
    parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        ):

    parameter_parser = parser.add_argument_group(
        title,
        "Parameters",
        gooey_options={'show_border': True, 'columns': 4})
    parameter_parser.add_argument(
        "--C", type=lambda x: eval(x),
        default=DEFAULT_PARAMS['C'],
        metavar="C",
        help="Penalty Parameter C of the error term"
    )
    parameter_parser.add_argument(
        "--kernel", choices=['linear', 'poly', 'rbf', 'sigmoid'],
        default=DEFAULT_PARAMS['kernel'],
        metavar="kernel type",
        help="Specifies the kernel type to be used in the algorithm. "
             "It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’."
    )
    parameter_parser.add_argument(
        "--degree", type=lambda x: eval(x),
        default=DEFAULT_PARAMS['degree'],
        metavar='ploynomial kernel degree',
        help="Degree of the polynomial kernel function (‘poly’)."
             "Ignored by all other kernels."
    )
    parameter_parser.add_argument(
        "--gamma", type=lambda x: x if x == 'auto' else eval(x),
        default=DEFAULT_PARAMS['gamma'],
        metavar="gamma",
        help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.",
    )
    parameter_parser.add_argument(
        "--coef0", type=lambda x: eval(x),
        default=DEFAULT_PARAMS['coef0'],
        metavar="coef0",
        help="Independent term in kernel function. "
             "It is only significant in ‘poly’ and ‘sigmoid’.",
    )

    show_and_save_parser = parser.add_argument_group(
        "",
        "Show and Save model options",
        gooey_options={'show_border': True, 'columns': 4})
    show_and_save_parser.add_argument(
        "--print-model-summary", action='store_true',
    )
    Path(MODEL_DIR).mkdir(exist_ok=True)
    show_and_save_parser.add_argument(
        "--save-path", type=str,
        metavar="File path",
        default="{}build_param.npz".format(MODEL_DIR),
        help="model name to save model",
        gooey_options={
            'validator': {
                'test': "user_input[:len('"+MODEL_DIR+"')]=='"+MODEL_DIR+"'",
                'message': 'unvalid save path'
            }
        }
    )
    show_and_save_parser.add_argument(
        "--save-file", type=str,
        metavar="Overwrite File",
        help="model name to save model",
        widget="FileChooser"
    )

    return parser


def build(args, baseModel=baseModel):
    base_model = baseModel(C=args.C,
                           kernel=args.kernel,
                           degree=args.degree,
                           gamma=args.gamma,
                           coef0=args.coef0,
                           )

    if args.print_model_summary:
        print(base_model)
    if args.save_path:
        with open(args.save_path, 'wb') as f:
            pickle.dump(args, f)
    if args.save_file:
        with open(args.save_path, 'wb') as f:
            pickle.dump(args, f)

    return args


if __name__ == "__main__":
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    print(args)
    parser._defaults['build'](args)


def save_model_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Save Model",
        description="",
        save_path="checkpoint/Xception/",
        save_name="model.h5"):
    if isinstance(parser, (ArgumentParser, GooeyParser)):
        save_parser = parser.add_argument_group(
            "Save Options", gooey_options={'columns': 2}
        )
    elif isinstance(parser, _ArgumentGroup):
        save_parser = parser
    else:
        raise ValueError
    save_parser.add_argument(
        "--save", action='store_true', default=False)
    # save_parser.add_argument(
    #     '--save_dir', type=str, default=save_path,
    #     metavar="Directory",
    #     help="directory to save model",
    #     # widget='DirChooser',
    # )
    save_parser.set_defaults(save_path=save_path)
    save_parser.add_argument(
        "save-name", type=str,
        default=save_name,
        metavar="Name",
        help="model name to save model",
    )

    def save_model(args):
        model = args.model
        model.save(parser._defaults["save_path"]+args.save_name)
        return model

    return parser


def save_load_parser(
        parser: Union[ArgumentParser, GooeyParser,
                      _ArgumentGroup] = GooeyParser(),
        title="Load Model",
        description="",
        load_path="checkpoint/Xception/",
        load_name="model.h5"):
    if isinstance(parser, (ArgumentParser, GooeyParser)):
        load_parser = parser.add_argument_group(
            "Save Options", gooey_options={'columns': 2}
        )
    elif isinstance(parser, _ArgumentGroup):
        load_parser = parser
    else:
        raise ValueError
    # save_parser.add_argument(
    #     '--save_dir', type=str, default=save_path,
    #     metavar="Directory",
    #     help="directory to save model",
    #     # widget='DirChooser',
    # )
    load_parser.set_defaults(load_path=load_path)
    load_parser.add_argument(
        "load-file", type=str,
        default=load_name,
        metavar="Model File",
        help="model name to save model",
    )

    def save_model(args):
        model = load_model(parser._defaults["load_path"]+args.load_name)
        return model

    return parser
