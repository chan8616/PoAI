from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import XceptionModel
from ..keras_applications import run as runlib
#  from ..keras_applications.run import run
from .config_samples import (XceptionTrainConfig,
                             XceptionImagenetConfig,
                             XceptionCIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:
    XceptionCIFAR10Config().display()
    return runlib.run_parser(parser,
                             title,
                             train_config=XceptionTrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', XceptionCIFAR10Config()),
                                 ('train_imagenet', XceptionImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(XceptionModel(), config)
    """
    def run(config):
        print(config)
        build_cmds, build_args, run_cmds, run_args, generator_cmds, generator_args, stream = config
        build_cmd = build_cmds[0]
        run_cmd = run_cmds[0]
        generator_cmd = generator_cmds[0]

        #  build_config = build_config.build_config(build_args)
        #  generator_config = generator_config.generator_config(generator_args)
        #  dataset, dataset_val = generator.generator(generator_cmd, generator_args)
        print('before generator')
        train_generator, val_generator = generator(generator_cmd, generator_args)

        print('before build')
        model = build.build(build_args)

        print('before load')
        if run_args.load_pretrained_weights is not None:
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
            train_args = run_args
            print('before train')
            #  model.train(train_args, train_generator, val_generator)
            return train.train((model,
                                train_args,
                                train_generator,
                                val_generator,
                                stream))
        elif 'test' == run_cmd:
            test_args = run_args
            now = datetime.datetime.now()
            result_dir = Path("{}{:%Y%m%dT%H%M}".format(
                    str(Path(test_args.result_path).parent), now))
            if not result_dir.exists():
                result_dir.mkdir(parents=True)
            model.result_dir = result_dir
            print('before test')
            return test.test(model, test_args, dataset)
    """
