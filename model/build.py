from typing import Union
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

from logistic.simple_logistic import build as simple_logistic_
from logistic.multilayer_logistic import build as multilayer_logistic_

from vgg.vgg16 import build as vgg16_
from vgg.vgg19 import build as vgg19_

from Xception import build as Xception_
from MobileNet import build as MobileNet_

def build_parser(
        parser: Union[ArgumentParser, GooeyParser] = GooeyParser(),
        title="Build Model",
        ):
    assert isinstance(parser, (ArgumentParser, GooeyParser)), type(parser)

    subs = parser.add_subparsers()

    simple_logistic_parser = subs.add_parser('simple_logistic')
    simple_logistic_.build_parser(simple_logistic_parser)

    multilayer_logistic_parser = subs.add_parser('multilayer_logistic')
    multilayer_logistic_.build_parser(multilayer_logistic_parser)

    vgg16_parser = subs.add_parser('vgg16')
    vgg16_.build_parser(vgg16_parser)
    # parser.set_defaults(vgg16=vgg16_parser._defaults['build'])

    vgg19_parser = subs.add_parser('vgg19')
    vgg19_.build_parser(vgg19_parser)
    # parser.set_defaults(vgg19=build)
    # vgg19_(vgg19_parser)
    # parser.set_defaults(vgg19=vgg19_parser._defaults['build'])

    Xception_parser = subs.add_parser('Xception')
    Xception_.build_parser(Xception_parser)

    MobileNet_parser = subs.add_parser('MobileNet')
    MobileNet_.build_parser(MobileNet_parser)

    return parser


def build(cmd, args):
    if 'simple_logistic' == cmd:
        simple_logistic_.build(args)
    elif 'multilayer_logistic' == cmd:
        multilayer_logistic_.build(args)
    elif 'vgg16' == cmd:
        vgg16_.build(args)
    elif 'vgg19' == cmd:
        vgg19_.build(args)
    elif 'Xception' == cmd:
        Xception_.build(args)
    elif 'MobileNet' == cmd:
        MobileNet_.build(args)
    else:
        raise NotImplementedError('wrong cmd:', cmd)


if __name__ == "__main__":
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    print(args)
    parser._defaults['build'](args)
