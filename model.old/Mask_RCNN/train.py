from typing import Union, Callable
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser
import imgaug

try:
    from . import build
except ImportError:
    import build


def train_setting_parser(
        parser: GooeyParser = GooeyParser(description="Train Options"),
        ) -> GooeyParser:

    build.build_parser(parser)

    train_setting_parser = parser.add_argument_group(
        'train setting',
        'Train Setting Options',
        gooey_options={'show_border': True, 'columns': 3})

    train_setting_parser.add_argument(
        "--epoch-schedule", type=eval, nargs=3, default='40, 120, 160',
        metavar='<train epoch schdule>',
        help="Epoch schedule per each training."
    )
    train_setting_parser.add_argument(
        "--validation_steps", type=int, default=None,
        help="number of steps (batches of samples) to validate before stopping"
    )
    train_setting_parser.add_argument(
        "--shuffle",
        action='store_true',
        default=True
    )

    return parser


def train_setting(args):
    return build.build(args)
    #  mode, ModelConfig, MODEL_DIR, args = build.build(args)

    #  Config = makeDatasetConfig(ModelConfig)
    #  config = Config()

    #  from . import modellib

    #  model = modellib.MaskRCNN(
    #          mode=mode,
    #          model_dir=MODEL_DIR,
    #          config=config)

    #  # Load weights
    #  print("Loading weights ", model_path)
    #  model.load_weights(args.save_path, by_name=True)


def train(train_setting, dataset_setting):
    # train setting
    mode, ModelConfig, MODEL_DIR, setting_args = train_setting

    # dataset
    CocoDataset, makeDatasetConfig, dataset_args = dataset_setting

    Config = makeDatasetConfig(ModelConfig)
    config = Config()

    from . import modellib

    model = modellib.MaskRCNN(
            mode=mode,
            model_dir=MODEL_DIR,
            config=config)

    # Load weights
    print("Loading weights ", setting_args.save_path)
    model.load_weights(setting_args.save_path, by_name=True)

    dataset_train = CocoDataset()
    dataset_train.load_coco(dataset_args.dataset,
                            "train",
                            year=dataset_args.year,
                            auto_download=False)
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_coco(dataset_args.dataset,
                          "val",
                          year=dataset_args.year,
                          auto_download=False)
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    # training
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=setting_args.epoch_schedule[0],
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=setting_args.epoch_schedule[1],
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=setting_args.epoch_schedule[2],
                layers='all',
                augmentation=augmentation)


def main():
    parser = Gooey(build_parser)()
    args = parser.parse_args()
    res = build(args)
    print(res)


if __name__ == '__main__':
    main()
