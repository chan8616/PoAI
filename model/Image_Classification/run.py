import warnings
warnings.simplefilter("ignore", UserWarning)

from gooey import Gooey, GooeyParser
import json
import sys
import dataset
import arguments
from predict import predict
from train import train_model
from utils import model_savefiles, tensorboard_savefiles, dataset_files, load_tfboard

save_dir = "./save_dir/image_classification"

@Gooey(image_dir='image/cv', optional_cols=2, program_name="Image Classification", default_size=(800,800), poll_external_updates = True)
def run():
    gooey_options={'show_border': True, 'columns': 2}

    parser = GooeyParser()
    subs = parser.add_subparsers(help='commands', dest='commands')
    dataset_parser = subs.add_parser('Dataset', help='Configurate Dataset')
    arguments.opt_transform(dataset_parser.add_argument_group("Image Transform option", gooey_options=gooey_options))
    arguments.opt_augment(dataset_parser.add_argument_group("Image augmentation option", gooey_options=gooey_options))
    arguments.opt_dataset(dataset_parser.add_argument_group("Dataset option", gooey_options=gooey_options))
    arguments.opt_split(dataset_parser.add_argument_group("Dataset split option", gooey_options=gooey_options))

    train_parser = subs.add_parser('Train', help='Configurate model training')
    dataset_group = train_parser.add_argument_group("Dataset option", gooey_options=gooey_options)
    arguments.load_dataset(dataset_group, dataset_files(save_dir, 'train'), 'train')
    arguments.load_dataset(dataset_group, dataset_files(save_dir, 'valid'), 'valid')
    arguments.load_dataset(dataset_group, dataset_files(save_dir, 'test'), 'test')
    arguments.opt_dataloader(train_parser.add_argument_group("Dataloader option", gooey_options=gooey_options))
    arguments.opt_model(train_parser.add_argument_group("Model Architecture option", gooey_options=gooey_options))
    arguments.opt_device(train_parser.add_argument_group("Device option", gooey_options=gooey_options))
    arguments.opt_train(train_parser.add_argument_group("Train option", gooey_options=gooey_options))
    arguments.opt_save(train_parser.add_argument_group("Save option", gooey_options=gooey_options))

    predict_parser = subs.add_parser('Prediction', help='Configurate model prediction')
    arguments.load_predictDataset(predict_parser.add_argument_group("Dataset option", gooey_options=gooey_options))
    arguments.opt_dataloader(predict_parser.add_argument_group("Dataloader option", gooey_options=gooey_options))
    arguments.load_model(predict_parser.add_argument_group("Model option", gooey_options=gooey_options), model_savefiles(save_dir))
    arguments.opt_device(predict_parser.add_argument_group("Device option", gooey_options=gooey_options))

    tfboard_parser = subs.add_parser('Tensorboard', help='Load TensorBoard')
    tfboard_group = tfboard_parser.add_argument_group("Tensorboard option", gooey_options=gooey_options)
    arguments.load_tfboard(tfboard_group, tensorboard_savefiles(save_dir))

    args = parser.parse_args()

    if args.commands =='Dataset':
        dataset.make(args, save_dir)
    elif args.commands =='Train':
        train_model(args, save_dir)
    elif args.commands == 'Prediction':
        predict(args, save_dir)
    elif args.commands == 'Tensorboard':
        load_tfboard(args, save_dir)

if 'gooey-seed-ui' in sys.argv:
    print(json.dumps({
    'dataset_train': dataset_files(save_dir, 'train'),
    'dataset_valid': dataset_files(save_dir, 'valid'),
    'datasetr_test': dataset_files(save_dir, 'test'),
    'modelLoad': model_savefiles(save_dir),
    'tensorboard': tensorboard_savefiles(save_dir)
    }))
else:
    run()