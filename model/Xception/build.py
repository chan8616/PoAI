from gooey import Gooey, GooeyParser

from . import train, test, generator


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()
    train_setting_parser = subs.add_parser('train')
    train.train_setting_parser(train_setting_parser)

    test_setting_parser = subs.add_parser('test')
    test.test_setting_parser(test_setting_parser)

    return parser


def build(build_cmd, build_args, generator_args):
    if 'train' == build_cmd:
        setting = train.train_setting(args)
        dataset = generator(generator_args)
        return train.train(setting, dataset)
    elif 'test' == build_cmd:
        setting = train.test_setting(args)
        dataset = generator(generator_args)
        return test.test(setting, dataset)
