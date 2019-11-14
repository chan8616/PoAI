import datetime
from pathlib import Path
from gooey import Gooey, GooeyParser

from .model import KerasAppBaseModel
from .train import TrainConfig, train_config_parser  # , train_config
from .generator import generator

#  from . import (train_config,
#                 build,
#                 build_config,
#                 train,
#                 config
#                 test, test_config,
#                 generator, generator_config,
#                 )


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        train_config=TrainConfig(),
        train_configs={},
        ) -> GooeyParser:

    subs = parser.add_subparsers()

    #  test_parser = subs.add_parser('test')
    #  test_config.test_config_parser(test_parser)

    for k, v in train_configs.items():
        train_parser = subs.add_parser(k)
        train_config_parser(train_parser, title, v)

    #  train_parser = subs.add_parser('train_check')
    #  train_config_parser(train_parser, title, train_imagenet_config)

    train_parser = subs.add_parser('train')
    train_config_parser(train_parser, title, train_config)

    #  train_parser = subs.add_parser('train_imagenet')
    #  train_config_parser(train_parser, title, train_imagenet_config)

    return parser


# Should be fixed. It is directly used in gui/frame.py
def run(model: KerasAppBaseModel, config):
    from .build import build
    from .train import train

    print(config)
    (build_cmds, build_args,
     run_cmds, run_args,
     generator_cmds, generator_args,
     stream) = config
    #  build_cmd = build_cmds[0]
    run_cmd = run_cmds[0]
    generator_cmd = generator_cmds[0]

    #  build_config = build_config.build_config(build_args)
    #  generator_config = generator_config.generator_config(generator_args)
    train_generator, val_generator = generator(generator_cmd, generator_args)

    #  model = build.build(build_args)
    #  model.build(build_args)
    print('before build')
    build(model, build_args)

    print('before load')
    if run_args.load_pretrained_weights:
        if run_args.load_pretrained_weights == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        elif run_args.load_pretrained_weights == "last":
            # Find last trained weights
            weights_path = model.find_last()
        else:
            weights_path = run_args.load_pretrained_weights

        model.load_weights(weights_path, by_name=True)
    print('load complete')

    if 'train' in run_cmd:
        print('before train')
        train_args = run_args
        #  model.train(train_args, train_generator, val_generator)
        train(model, train_args, train_generator, val_generator, stream)
    elif 'test' == run_cmd:
        test_args = run_args
        now = datetime.datetime.now()
        result_dir = Path("{}{:%Y%m%dT%H%M}".format(
                str(Path(test_args.result_path).parent), now))
        if not result_dir.exists():
            result_dir.mkdir(parents=True)
    del model
    del train_generator, val_generator
        #  model.result_dir = result_dir
        #  print('before test')
        #  return test.test(model, test_args, dataset)
    #      setting = train.test_setting(model, run_args)
    #      dataset = generator(generator_args)
    #      return test.test(setting, dataset)
