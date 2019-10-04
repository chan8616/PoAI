from typing import Union, Callable
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from keras.models import load_model

from model.utils.callbacks \
    import csvlogger_callback_parser, get_csvlogger_callback
from model.utils.callbacks_ import MyCSVLogger


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
        '--use-csvlogger-callback', default=False,
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

    for callback in callbacks:
        callback.model = model
        callback.model.stop_training = False
        if isinstance(callback, MyCSVLogger):
            callback.on_test_begin()

    for x, y in test_generator:
        print(test_generator.batch_index, test_generator.total_batches_seen)
        if (test_generator.batch_index != test_generator.total_batches_seen):
            break
        logs = {'batch': test_generator.batch_index,
                'size': test_generator.batch_size}

        try:
            outputs = model.test_on_batch(x, y)
            logs.update({'loss': outputs})
        except:
            outputs = model.predict_on_batch(x)
            logs.update({'outputs': outputs})

        for callback in callbacks:
            if isinstance(callback, MyCSVLogger):
                callback.on_test_batch_end(test_generator.batch_index, logs)

    for callback in callbacks:
        if isinstance(callback, MyCSVLogger):
            callback.on_test_end()

    ### released keras version (2.2.4) not support callbacks in eval, pred.
    # model.evaluate_generator(test_generator,
    #                          callbacks=callbacks,
    #                          )

    # results = model.evaluate_generator(test_generator,
    #                                    steps=1,
    #                                    # callbacks=callbacks,
    #                                    )
    # print('{}: {}'.format(model.metrics_names, results))

    # predictions = model.predict_generator(test_generator,
    #                                       steps=1,
    #                                       )

    # if args2
    # import pandas as pd
    # df = pd.DataFrame(predictions)
    # df.to_csv(callbacks[0].filename, index=False)


def main():
    from generator.image_classification.image_generator \
        import image_generator_parser
    # parser = Gooey(callbacks_parser)()
    model_parser = Gooey(test_setting_parser)()
    model_args = model_parser.parse_args()
    data_parser = Gooey(image_generator_parser)()
    data_args = data_parser.parse_args()
    print(model_args)
    # print(data_ags)

    args1 = test_setting(model_args)
    args2 = data_parser._defaults['generator'](data_args)
    print(args1)
    print(args2)
    test(args1, args2)

if __name__ == "__main__":
    main()
