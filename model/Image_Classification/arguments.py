from gooey import GooeyParser
import torch

model_vgg=['VGG11','VGG11 with batch normalization', 'VGG13','VGG13 with batch normalization', 'VGG16','VGG16 with batch normalization', 'VGG19','VGG19 with batch normalization',]
model_res=[ 'ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152',]
model_dense=['DenseNet-121', 'DenseNet-161', 'DenseNet-169', 'DenseNet-201',]
model_squeeze=[ 'SqueezeNet 1.0', 'SqueezeNet 1.1',]
model_others = ['AlexNet', 'Inception v3', 'GoogLeNet', 'ShuffleNet v2', 'MobileNet v2', 'MNASNet 1.0', 'ResNeXt-50-32x4d', 'ResNeXt-101-32x8d', 'Wide ResNet-50-2', 'Wide ResNet-101-2',]
optimizers = ['Adam', 'RMSprop', 'Adagrad', 'SGD', 'Adadelta', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'Rprop'] 

########################## dataset and dataloader option ##########################

def opt_transform(parser: GooeyParser = GooeyParser()):
    parser.add_argument( "--gray",
                        # metavar="Grayscale",
                        metavar="Grayscale image",
                        action='store_true',
                        )

    parser.add_argument(
        "--resize", type=int,
        metavar='Resize image',
        default=0,
        # help="Resize the image to the given size",
        gooey_options={
            'validator': {
                'test': 'int(user_input) >= 0',
                'message': 'Must be non-negative integer.'
            }
        }
    )

    parser.add_argument(
        "--center_crop", type=int,
        metavar='Center crop size',
        default=0,
        help="Crop the image at the center",
        gooey_options={
            'validator': {
                'test': 'int(user_input) >= 0',
                'message': 'Must be non-negative integer.'
            }
        }
    )

def opt_augment(parser: GooeyParser = GooeyParser()):
    parser.add_argument( "--hflip",
                        metavar="Horizontal flip",
                        # help="Horizontally flip the given image randomly",
                        action='store_true',
                        )

    parser.add_argument( "--vflip",
                        metavar="Vertical flip",
                        # help="Vertically flip the given image randomly",
                        action='store_true',
                        )

    parser.add_argument(
        "--random_crop", type=int,
        metavar='Image random crop',
        default=0,
        help="Crop the image at a random location",
        gooey_options={
            'validator': {
                'test': 'int(user_input) >= 0',
                'message': 'Must be non-negative integer.'
            }
        }
    )

    parser.add_argument(
        "--random_rot", type=float,
        metavar='Image random rotation',
        default=0,
        help="Rotate the image by angle randomly\n(range : [-degree, +degree]",
        gooey_options={
            'validator': {
                'test': 'int(user_input) >= 0',
                'message': 'Must be non-negative number.'
            }
        }
    )
    parser.add_argument("--save_sample",
                        metavar="Save sample images",
                        action='store_true',
                        )

    # parser.add_argument(
    #     "--random_bright", type=float,
    #     metavar='Jitter brightness',
    #     default=0,
    #     help="Randomly change brightness(default : Don't)",
    #     gooey_options={
    #         'validator': {
    #             'test': 'int(user_input) > 0',
    #             'message': 'Must be positive number.'
    #         }
    #     }
    # )
    #
    # parser.add_argument(
    #     "--random_contrast", type=float,
    #     metavar='Jitter contrast',
    #     default=0,
    #     help="Randomly change contrast(default : Don't)",
    #     gooey_options={
    #         'validator': {
    #             'test': 'int(user_input) > 0',
    #             'message': 'Must be positive number.'
    #         }
    #     }
    # )
    # parser.add_argument(
    #     "--random_saturation", type=float,
    #     metavar='Jitter saturation',
    #     default=0,
    #     help="Randomly change saturation(default : Don't)",
    #     gooey_options={
    #         'validator': {
    #             'test': 'int(user_input) > 0',
    #             'message': 'Must be positive number.'
    #         }
    #     }
    # )
    # parser.add_argument(
    #     "--random_hue", type=float,
    #     metavar='Jitter Hue',
    #     default=0,
    #     help="Randomly change hue(default : Don't)",
    #     gooey_options={
    #         'validator': {
    #             'test': 'int(user_input) > 0',
    #             'message': 'Must be positive number.'
    #         }
    #     }
    # )

def opt_dataset(parser: GooeyParser = GooeyParser()):
    dataset_kind = parser.add_mutually_exclusive_group()
    dataset_kind.add_argument('--user_dataset',
                            widget='DirChooser',
                            dest = "Your Dataset")

    dataset_kind.add_argument('--example_dataset',
                            choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'],
                            dest="Prepared Dataset")

def opt_split(parser: GooeyParser = GooeyParser()):
    parser.add_argument("--split", type=float,
                        help="Ratio of train data among all",
                        default=0.8,
                        metavar="Train data ratio",
                        gooey_options={
                            'validator': {
                                'test': '0<=float(user_input)<=1',
                                'message': 'Must be between 0 and 1.'
                            }
                        }
                        )
    parser.add_argument(
        "--valid_split", type=float,
        default=0.2,
        help="Ratio of validation data among train data",
        metavar="Validation data ratio",
        gooey_options={
            'validator': {
                'test': '0<=float(user_input)<=1',
                'message': 'Must be between 0 and 1.'
            }
        }
    )

def opt_dataloader(parser: GooeyParser = GooeyParser(), train =True):
    parser.add_argument(
        "--batch_size", type=int,
        metavar='Batch size',
        default=32,
        # help="How many samples per batch to load",
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive integer.'
            }
        }
    )
    
    if train == True:
        parser.add_argument("--shuffle",
                            metavar="Shuffle",
                            # help="Set this option to have the data reshuffled at every epoch",
                            action='store_true',
                            )

########################## train option and model build ##########################
def opt_model(parser: GooeyParser = GooeyParser()):
    model_option = parser.add_mutually_exclusive_group()
    model_option.add_argument(
        "--model-vgg", type=str,
        choices=model_vgg,
        metavar='VGGNet',
    )
    model_option.add_argument(
        "--model-res", type=str,
        choices=model_res,
        metavar='ResNet',
    )
    model_option.add_argument(
        "--model-des", type=str,
        choices=model_dense,
        metavar='DenseNet',
    )
    model_option.add_argument(
        "--model-squ", type=str,
        choices=model_squeeze,
        metavar='SqueezeNet',
    )
    model_option.add_argument(
        "--model-oth", type=str,
        choices=model_others,
        metavar='Ohter Networks',
    )

def opt_train(parser: GooeyParser = GooeyParser()):
    parser.add_argument( "--preTrain",
                        default = False,
                        metavar="Finetue pretrained model",
                        action='store_true',)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=optimizers,
        metavar='optimizer',
    )
    parser.add_argument( "--lr", type=float,
                        metavar="learning rate",
                        default = 0.001,
                        gooey_options={
                            'validator': {
                                'test': '0<=float(user_input)<=1',
                                'message': 'Must be between 0 and 1.'
                            }
                        }
                        )
    parser.add_argument(
        "--epoch", type=int,
                        metavar="epoch",
                        default = 10,
                        gooey_options={
                            'validator': {
                                'test': '0<int(user_input)',
                                'message': 'Must be positive integer.'
                            }
                        }
                        )

def opt_device(parser: GooeyParser = GooeyParser()):
    device_option = parser.add_mutually_exclusive_group()
    device_option.add_argument( "--cpu", metavar="CPU",
                        action='store_true',)

    n_gpus = torch.cuda.device_count()
    available_gpu_text = 'Available GPU number: '
    for i in range(n_gpus):
        available_gpu_text += str(i) + ' '

    device_option.add_argument( "--gpu", type =str,
                        help =available_gpu_text, metavar="GPU",
                        gooey_options={
                            'validator': {
                                'test': 'all([e.isdigit() for e in user_input.replace(" ","").split(",")])',
                                'message': 'Gpu device number must be positive integer.'
                            }
                        })

def opt_save(parser: GooeyParser = GooeyParser()):
    parser.add_argument(
        "--tfboard",
        default = True,
        metavar="Save Tensorbard",
        action='store_true')

    parser.add_argument( "--save_best",
                        metavar="Save best only",
                        action='store_true')


############################## load option ##############################

def load_model(parser: GooeyParser = GooeyParser(), model_savefiles=[]):
    parser.add_argument('modelLoad',
                        metavar='Load Model',
                        widget='Dropdown',
                        choices=model_savefiles,
                        gooey_options={
                            'validator': {
                                'test': 'user_input != "Select Option"',
                                'message': 'Choose a save file from the list'}})

def load_dataset(parser: GooeyParser = GooeyParser(), dataset_files=[], dset='train'):
    parser.add_argument('dataset_'+dset,
                        metavar='Load {} set'.format(dset),
                        widget='Dropdown',
                        choices=dataset_files,
                        gooey_options={
                            'validator': {
                                'test': 'user_input != "Select Option"',
                                'message': 'Choose a save file from the list'}})


def load_tfboard(parser: GooeyParser = GooeyParser(), tensorboard_files=[]):
    parser.add_argument('tensorboard',
                        metavar='Load Tensorbard',
                        widget='Dropdown',
                        choices=tensorboard_files,
                        gooey_options={
                            'validator': {
                                'test': 'user_input != "Select Option"',
                                'message': 'Choose a save file from the list'}})

def load_predictDataset(parser: GooeyParser = GooeyParser()):
    parser.add_argument('--predict_dataset',
                            widget='DirChooser',
                            metavar='Your Dataset')
