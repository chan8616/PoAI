from typing import Union, Callable
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from keras.models import load_model

from model.utils.callbacks \
    import csvlogger_callback_parser, get_csvlogger_callback


def get_callbacks_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title: str = "Callbacks Setting",
        ) -> Callable:

    csvlogger_parser = parser.add_argument_group(
        title=title,
        description="csvlogger callback",
        gooey_options={'columns': 3, 'show_border': True}
    )
    csvlogger_parser.add_argument(
        '--use-csvlogger-callback',
        action='store_true',
    )
    csvlogger_callback_parser(csvlogger_parser)

    return parser


def get_callbacks(args):
    callbacks = []
    if args.use_csvlogger_callback:
        callbacks.append(
            get_csvlogger_callback(args))

    return callbacks


def test_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Test Setting",
        description="") -> Callable:

    test_setting_parser = parser.add_argument_group(
        title=title,
        description=description)

    test_setting_parser.add_argument(
        'load_file',
        widget='FileChooser'
    )

    get_callbacks_parser(parser)

    return parser


def test_setting(args):
    model = load_model(args.load_file)
    return (model,
            get_callbacks(args))


def test(args1, args2):
    model, callbacks = args1
    test_generator, _ = args2
    # model.evaluate_generator(test_generator,
    #                          callbacks=callbacks,
    #                          )

    results = model.evaluate_generator(test_generator,
                                       steps=1,
                                       # callbacks=callbacks,
                                       )
    print('{}: {}'.format(model.metrics_names, results))

    predictions = model.predict_generator(test_generator,
                                          steps=1,
                                          )
    import pandas as pd
    df = pd.DataFrame(predictions)
    df.to_csv(callbacks[0].filename, index=False)


if __name__ == "__main__":
    from generator.image_classification.image_generator \
        import image_generator_parser
    # parser = Gooey(callbacks_parser)()
    model_parser = Gooey(test_setting_parser)()
    model_args = model_parser.parse_args()
    data_parser = Gooey(image_generator_parser)()
    data_args = data_parser.parse_args()
    print(model_args)
    # print(data_ags)

    args1 = model_parser._defaults['test_setting'](model_args)
    args2 = data_parser._defaults['generator'](data_args)
    print(args1)
    print(args2)
    test(args1, args2)
