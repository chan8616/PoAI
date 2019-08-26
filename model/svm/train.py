from typing import Union, Callable
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from model.svm import build

DEFAULT_PARAMS={'tol': 1e-3,
                'max_iter': -1,
                'dicision_function_shape': 'ovr'}

def train_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Train Setting",
        description="") -> Callable:

    parser.add_argument(
        'load_build_params',
        widget='FileChooser'
    )

    parameter_parser = parser.add_argument_group(
        title,
        "Parameters",
        gooey_options={'show_border': True, 'columns': 4})
    parameter_parser.add_argument(
        "--tol", type=lambda x: eval(x),
        default=DEFAULT_PARAMS['tol'],
        metavar="tolerance",
        help="Tolerance for stopping criterion.",
    )
    parameter_parser.add_argument(
        "--class_weight", choices=['balanced'],
        metavar="class weight",
        help="Set the parameter C of class i to class_weight[i]*C for SVC. "
             "If not given, all classes are supposed to have weight one. "
             "The “balanced” mode uses the values of y to automatically adjust weights "
             "inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))",
    )
    parameter_parser.add_argument(
        "--max_iter", type=lambda x: eval(x),
        default=DEFAULT_PARAMS['max_iter'],
        metavar='max iteration',
        help="Hard limit on iterations within solver, or -1 for no limit.",
    )
    parameter_parser.add_argument(
        "--decision_function_shape", choices=['ovo', 'ovr'],
        default=DEFAULT_PARAMS['dicision_function_shape'],
        metavar="dicision_function_shape",
        help="Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, "
             "or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). "
             "However, one-vs-one (‘ovo’) is always used as multi-class strategy.",
    )

    return parser


def train_setting(args):
    print(args)
    return (model, args.epochs,
            args.epochs if args.validation_steps is None else args.validation_steps,
            get_callbacks(args), args.shuffle)


def train(args1, args2):
    model, epochs, validation_steps, callbacks, shuffle = args1
    train_generator, validation_generator = args2
    model.fit_generator(train_generator,
                        epochs,
                        callbacks=callbacks,
                        # validation_split=validation_split,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        shuffle=shuffle,
                        # initial_epoch=initial_epoch,
                        # steps_per_epoch=steps_per_epoch,
                        )


"""
def train_parser(
        train_setting_parser: Union[ArgumentParser, GooeyParser,
                                    _ArgumentGroup] = GooeyParser(),
        data_generator_parser: Union[ArgumentParser, GooeyParser,
                                     _ArgumentGroup] = GooeyParser(),
        # parser: Union[ArgumentParser, GooeyParser,
        #               _ArgumentGroup] = GooeyParser(),
        title="Train Model",
        description=""):

    parser = ArgumentParser()
    # train_setting_parser(parser)
    # image_generator_parser(parser)

    # def train(model, callbacks,  # train setting output
    #           epochs, initial_epoch,
    #           steps_per_epoch, validation_steps,
    #           train_data, val_data,      # data generator output
    #           validation_split,
    #           shuffle,
    #           ):

    def train(train_setting_args, dataset_generator_args):
        model, epochs, callbacks, shuffle = train_setting_parser._defaults[
            'train_setting'](train_setting_args)

        train_generator, validation_generator = \
            dataset_generator_args._defaults[
                'generator'](dataset_generator_args)
        model.fit_generator(train_generator,
                            epochs,
                            callbacks=callbacks,
                            # validation_split=validation_split,
                            validation_data=validation_generator,
                            shuffle=shuffle,
                            # initial_epoch=initial_epoch,
                            # steps_per_epoch=steps_per_epoch,
                            # validation_steps=validation_steps,
                            )
        return
    parser.set_defaults(train=train)
    return parser
"""

if __name__ == "__main__":
    # from image_classification import flow_from_dirctory_parser
    # from image_classification import image_preprocess
    from generator.image_classification.image_generator \
        import image_generator_parser

    # parser = Gooey(callbacks_parser)()
    model_parser = Gooey(train_setting_parser)()
    model_args = model_parser.parse_args()
    data_parser = Gooey(image_generator_parser)()
    data_args = data_parser.parse_args()
    print(model_args)
    # print(data_ags)

    args1 = model_parser._defaults['train_setting'](model_args)
    args2 = data_parser._defaults['generator'](data_args)
    print(args1)
    print(args2)
    train(args1, args2)
