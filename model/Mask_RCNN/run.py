from gooey import Gooey, GooeyParser

from . import (build, build_config,
               train, train_config,
               generator, generator_config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        description="") -> GooeyParser:

    train.train_parser(parser)

    #  subs = parser.add_subparsers()

    #  train_config_parser = subs.add_parser('train')
    #  train.train_config_parser(train_config_parser)

    #  test_setting_parser = subs.add_parser('test')
    #  test.test_setting_parser(test_setting_parser)

    return parser


# Should be fixed. It is directly used in gui/frame.py
def run(build_cmds, build_args,
        run_cmds, run_args,
        generator_cmds, generator_args):
    build_cmd = build_cmds[0]
    run_cmd = run_cmds[0]
    generator_cmd = generator_cmds[0]

    #  build_config = build_config.build_config(build_args)
    #  generator_config = generator_config.generator_config(generator_args)
    dataset_train, dataset_val = generator.generator(
            generator_cmd, generator_args)

    if 'train' in run_cmd:
        train_args = run_args
        #  train_config = train_config.train_config(run_args)
        model = build.build('training',
                            build_args,
                            build_config.build_config(build_args),
                            run_args,
                            train_config.train_config(run_args),
                            generator_config.generator_config(generator_args),
                            )

        print('before load')
        if train_args.load_pretrained_weights == "coco":
            weights_path = model.get_coco_weights()
        elif train_args.load_pretrained_weights == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif train_args.load_pretrained_weights == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = train_args.load_pretrained_weights
        model.load_weights(weights_path)

        #  config = model_config(build_config, train_config),
        #  model = modellib.MaskRCNN(model=run_cmd,
        #                            config=config,
        #                            model_dir=MODEL_DIR.joinpath(args.save_dir))
        #  setting = train.train_setting(model, run_args)
        print('before train')
        return train.train(model, train_args, dataset_train, dataset_val)
    #  elif 'test' == build_cmd:
    #      setting = train.test_setting(model, run_args)
    #      dataset = generator(generator_args)
    #      return test.test(setting, dataset)
