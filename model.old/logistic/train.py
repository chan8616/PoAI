from pathlib import Path
from typing import Union

from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from simple_logistic.train import train_setting_parser as simple_
from multilayer_logistic.train import train_setting_parser as multi_
# from dataset.parser import generator


def train_setting_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        ) -> Union[ArgumentParser, GooeyParser]:
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)
    subs = parser.add_subparsers()

    simple_parser = subs.add_parser('simple_logistic')
    train_setting = simple_(simple_parser)
    parser.set_defaults(simple_logistic=train_setting)

    multi_parser = subs.add_parser('multilayer_logistic')
    train_setting = multi_(multi_parser)
    parser.set_defaults(multilayer_logistic=train_setting)

    return parser


if __name__ == "__main__":
    # parser = Gooey(callbacks_parser)()
    parser = Gooey(train_setting_parser)()
    # parser = Gooey(train_parser)()
    args = parser.parse_args()
    print(args)


# def train_parser(
#         parser: Union[ArgumentParser, GooeyParser,
#                       _ArgumentGroup] = GooeyParser(),
#         title="Train Model",
#         description=""):

#     # modelLoadParser(parser, )
#     # compile_parser(parser)
#     train_setting_parser(parser)
#     image_generator(parser)
#     # parser.set_defaults(train=train)

#     def train(model, callbacks,  # train setting output
#               epochs, initial_epoch,
#               steps_per_epoch, validation_steps,
#               train_data, val_data,      # data generator output
#               validation_split,
#               shuffle,
#               ):
#         train_data = args.train_data
#         val_data = args.val_data

#         model.fit_generator(*train_data,
#                             epochs,
#                             callbacks=callbacks,
#                             validation_split=validation_split,
#                             validation_data=val_data,
#                             shuffle=shuffle,
#                             initial_epoch=initial_epoch,
#                             steps_per_epoch=steps_per_epoch,
#                             validation_steps=validation_steps,
#                             )
#         return

#     return parser
