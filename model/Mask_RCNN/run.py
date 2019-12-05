import datetime
from pathlib import Path
from gooey import Gooey, GooeyParser

from . import (build, build_config,
               train, train_config,
               test, test_config,
               generator, generator_config,
               )
#  from generator.image.image_preprocess.image_preprocess_mask_rcnn import (
#          generator, generator_config) as (a, b)

from .config_samples import (BalloonConfig, CocoConfig,
                             NucleusConfig, ShapesConfig)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    test_parser = subs.add_parser('test')
    test_config.test_config_parser(test_parser)

    balloon_train_parser = subs.add_parser('train_balloon')
    train_config.train_config_parser(balloon_train_parser,
                                     BalloonConfig(),)

    train_parser = subs.add_parser('train')
    train_config.train_config_parser(train_parser)

    return parser


# Should be fixed. It is directly used in gui/frame.py
def run(config):
    print(config)
    build_cmds, build_args, run_cmds, run_args, generator_cmds, generator_args, stream = config
    build_cmd = build_cmds[0]
    run_cmd = run_cmds[0]
    generator_cmd = generator_cmds[0]

    #  build_config = build_config.build_config(build_args)
    #  generator_config = generator_config.generator_config(generator_args)
    dataset, dataset_val = generator.generator(generator_cmd, generator_args)

    if 'train' in run_cmd:
        print('train build')
        train_args = run_args
        #  train_config = train_config.train_config(run_args)
        model = build.build('training',
                            build_args,
                            build_config.build_config(build_args),
                            run_args,
                            train_config.train_config(run_args),
                            generator_config.generator_config(generator_args),
                            )
    elif 'test' == run_cmd:
        print('test build')
        test_args = run_args
        model = build.build('inference',
                            build_args,
                            build_config.build_config(build_args),
                            run_args,
                            test_config.test_config(run_args),
                            generator_config.generator_config(generator_args),
                            )
    else:
        raise AttributeError("run_cmd must be train or test!")

    print('before load')
    if run_args.load_pretrained_weights == "coco":
        weights_path = model.get_coco_weights()
    elif run_args.load_pretrained_weights == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif run_args.load_pretrained_weights == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = run_args.load_pretrained_file

    if 'coco' not in run_cmd and \
            run_args.load_pretrained_weights == "coco":
        print('load coco in balloon model')
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        print('load balloon model')
        model.load_weights(weights_path, by_name=True)
    print('load complete')

    #  config = model_config(build_config, train_config),
    #  model = modellib.MaskRCNN(model=run_cmd,
    #                            config=config,
    #                            model_dir=MODEL_DIR.joinpath(args.save_dir))
    #  setting = train.train_setting(model, run_args)
    if 'train' in run_cmd:
        print('before train')
        return train.train((model, train_args, dataset, dataset_val, stream))
    elif 'test' == run_cmd:
        now = datetime.datetime.now()
        result_dir = Path("{}{:%Y%m%dT%H%M}".format(
                str(Path(test_args.result_path).parent), now))
        if not result_dir.exists():
            result_dir.mkdir(parents=True)
        model.result_dir = result_dir
        print('before test')
        return test.test(model, test_args, dataset, stream)
    #      setting = train.test_setting(model, run_args)
    #      dataset = generator(generator_args)
    #      return test.test(setting, dataset)
