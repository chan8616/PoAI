from pathlib import Path
from typing import Union
from gooey import Gooey, GooeyParser

import model as modellib
from .build_config import build_config_parser, build_config, BuildConfig


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        build_config=BuildConfig(),
        imagenet_config=BuildConfig(),
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    base_build_parser = subs.add_parser('build_check')
    build_config_parser(base_build_parser, title, imagenet_config)

    build_parser = subs.add_parser('build')
    build_config_parser(build_parser, title, build_config)

    base_build_parser = subs.add_parser('build_imagenet')
    build_config_parser(base_build_parser, title, imagenet_config)

    return parser


def build(model, build_args) -> None:
    model.build(build_config=build_config(build_args))

    if build_args.print_model_summary:
        print(model.keras_model.summary())

    """
    build_config = build_config(build_args)

    x = base_model.output
    if 'flatten' == build_config.POOLING:
        x = Flatten(name='flatten')(x)
    elif 'avg' == build_config.POOLING:
        x = GlobalAveragePooling2D('avg_pool')(x)
    elif 'max' == build_config.POOLING:
        x = GlobalMaxPooling2D('max_pool')(x)

    for i, nodes in enumerate(build_config.HIDDEN_LAYERS):
        x = Dense(nodes, activation='relu',
                  name='fc{}'.format(i))(x)
    x = Dense(build_config.CLASSES, activation='softmax')(x)

    model = Model(
        inputs=base_model.input,
        outputs=x,
        name=build_config.NAME)

    if build_args.print_model_summary:
        model.summary()

    return model
    """
#  def build(base_model, build_args):
#      model = modellib.KerasAppBaseModel(
#              base_model = None,
#              config=build_config(build_args),
#              model_dir=str(log_dir))

#      if build_args.print_model_summary:
#          model.keras_model.summary()

#      return model
