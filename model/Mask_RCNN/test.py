import sys
from pathlib import Path
from gooey import Gooey, GooeyParser


try:
    from . import build
except ImportError:
    import build


def test_setting_parser(
        parser: GooeyParser = GooeyParser(description="Test Options"),
        ) -> GooeyParser:

    build.build_parser(parser)

    text_parser = parser.add_argument_group(
            'text',
            'Text Options',
            gooey_options={'show_border': True, 'columns': 4},
            )
    text_parser.add_argument(
            '--save-excel-result',
            metavar='Save text result: <True|False>',
            action='store_true',
            default=True,
            )
    text_parser.add_argument(
            '--excel-file',
            metavar='Text file: /path/to/excel.xlsx',
            help="Path to xlsx file to save output values.",
            default="logs/Mask_RCNN/result.xlsx",
            )

    image_parser = parser.add_argument_group(
            'image',
            'Image Options',
            gooey_options={'show_border': True, 'columns': 4},
            )
    image_parser.add_argument(
            '--save-image-result',
            metavar='Save image result: <True|False>',
            action='store_true',
            default=True,
            )
    image_parser.add_argument(
            '--image-directory',
            metavar='Image directory: /path/to/image/',
            help="Directory to save result image.",
            default="logs/Mask_RCNN/",
            )

    return parser


def test_setting(args):
    return build.build(args)


def test(test_setting, dataset_setting):
    #  print(test_setting)
    #  print(dataset_setting)
    mode, ModelConfig, MODEL_DIR, args = test_setting
    dataset, makeDatasetConfig = dataset_setting
    #  print('args', args)

    #  class Config(DatasetConfig, ModelConfig):
    Config = makeDatasetConfig(ModelConfig)
    config = Config()
    print(config.display())

    from . import modellib

    model = modellib.MaskRCNN(
            mode=mode,
            model_dir=MODEL_DIR,
            config=config)
    file_names, images = dataset
    file_names, images = (list(file_names), list(images))
    dataset = (file_names, images)

    results = []
    for file_name, image in zip(*dataset):
        try:
            results += model.detect([image])
        except ValueError:
            print("{} is not loaded.".format(file_name))
            file_names.remove(file_name)
            images.remove(image)
            continue

        if args.save_image_result:
            from . import visualize
            from matplotlib import pyplot as plt

            Path(args.image_directory).mkdir(exist_ok=True, parents=True)
            result = results[-1]
            fig, ax = plt.subplots(1, figsize=image.shape[:2])

            visualize.display_instances(image, result['rois'],
                                        result['masks'], result['class_ids'],
                                        config.CLASS_NAMES, result['scores'],
                                        ax=ax)
            # TODO: savefig segmentatil fault
            #  fig.savefig(args.image_directory + file_name)
            #  try:
            #      fig.savefig(Path(args.image_directory).joinpath(file_name))
            #  except Exception as e:
            #      print(e)
            #      break

        if args.save_excel_result:
            import pandas as pd
            df = pd.DataFrame(results)
            Path(args.excel_file).parent.mkdir(exist_ok=True, parents=True)
            df.to_excel(args.excel_file, engine='xlsxwriter')


if __name__ == "__main__":
    parser = Gooey(test_setting_parser)()
    import sys
    print(sys.argv[0])
    args = parser.parse_args()
    #  print(test_setting(args))
