import torch
import torch.nn as nn
from torch import optim
from torchvision import models

def get_model(model_name, in_ch, num_classes, pretrained):
    if in_ch == 1:
        gray = True
    elif in_ch == 3:
        gray = False
    ################ SqueezeNet ################
    if(model_name == 'SqueezeNet 1.0'):
        model = models.squeezenet1_0(pretrained = pretrained)
    if(model_name == 'SqueezeNet 1.1'):
        model = models.squeezenet1_1(pretrained = pretrained)

    ################ VGG ################
    if(model_name == 'VGG11'):
        model = models.vgg11(pretrained = pretrained)
    if(model_name == 'VGG11 with batch normalization'):
        model = models.vgg11_bn(pretrained = pretrained)
    if(model_name == 'VGG13'):
        model = models.vgg13(pretrained = pretrained)
    if(model_name == 'VGG13 with batch normalization'):
        model = models.vgg13_bn(pretrained = pretrained)
    if(model_name == 'VGG16'):
        model = models.vgg16(pretrained = pretrained)
    if(model_name == 'VGG16 with batch normalization'):
        model = models.vgg16_bn(pretrained = pretrained)
    if(model_name == 'VGG19'):
        model = models.vgg19(pretrained = pretrained)
    if(model_name == 'VGG19 with batch normalization'):
        model = models.vgg19_bn(pretrained = pretrained)
        
    ################ ResNet ################
    if(model_name == 'ResNet-18'):
        model = models.resnet18(pretrained = pretrained)
    if(model_name == 'ResNet-34'):
        model = models.resnet34(pretrained = pretrained)
    if(model_name == 'ResNet-50'):
        model = models.resnet50(pretrained = pretrained)
    if(model_name == 'ResNet-101'):
        model = models.resnet101(pretrained = pretrained)
    if(model_name == 'ResNet-152'):
        model = models.resnet152(pretrained = pretrained)

    ################ DenseNet ################
    if(model_name == 'DenseNet-121'):
        model = models.densenet121(pretrained = pretrained)
    if(model_name == 'DenseNet-161'):
        model = models.densenet161(pretrained = pretrained)
    if(model_name == 'DenseNet-169'):
        model = models.densenet169(pretrained = pretrained)
    if(model_name == 'DenseNet-201'):
        model = models.densenet201(pretrained = pretrained)

    ################ Other Networks ################
    if(model_name == 'AlexNet'):
        model = models.alexnet(pretrained = pretrained)
    if(model_name == 'Inception v3'):
        model = models.inception_v3(pretrained = pretrained)
    if(model_name == 'GoogLeNet'):
        model = models.googlenet(pretrained = pretrained)
    if(model_name == 'ShuffleNet v2'):
        model = models.shufflenet_v2_x1_0(pretrained = pretrained)
    if(model_name == 'MobileNet v2'):
        model = models.mobilenet_v2(pretrained = pretrained)
    if(model_name == 'MNASNet 1.0'):
        model = models.mnasnet1_0(pretrained = pretrained)
    if(model_name == 'ResNeXt-50-32x4d'):
        model = models.resnext50_32x4d(pretrained = pretrained)
    if(model_name == 'ResNeXt-101-32x8d'):
        model = models.resnext101_32x8d(pretrained = pretrained)
    if(model_name == 'Wide ResNet-50-2'):
        model = models.wide_resnet50_2(pretrained = pretrained)
    if(model_name == 'Wide ResNet-101-2'):
        model = models.wide_resnet101_2(pretrained = pretrained)

    if ('VGG' in model_name) or ('Dense' in model_name) or ('Squeeze' in model_name) or (model_name in ['AlexNet', 'MobileNet v2', 'MNASNet 1.0']):
        if pretrained and (gray == False):  #layer freeze(unless gray scale iamge)
            for layer in model.features.parameters():
                layer.requires_grad = False
        # layer change
        if gray:
            out_channels = model.features[0].out_channels
            kernel_size = model.features[0].kernel_size[0]
            stride = model.features[0].stride[0]
            padding = model.features[0].padding[0]
            model.features[0] = nn.Conv2d(1, out_channels, kernel_size, stride, padding)
        
        if 'Squeeze' in model_name:
            in_channels = model.classifier[1].in_channels
            model.classifier[1] = nn.Conv2d(in_channels, num_classes)
        else:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        if pretrained and (gray == False): #layer freeze(unless gray scale iamge)
            for layer in model.parameters():
                layer.requires_grad = False
        # last layer change
        if gray:
            out_channels = model.conv1.out_channels
            kernel_size = model.conv1.kernel_size[0]
            stride = model.conv1.stride[0]
            padding = model.conv1.padding[0]
            model.conv1 = nn.Conv2d(1, out_channels, kernel_size, stride, padding)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    print('\nLoad model :', model_name)
    print(model,'\n')
    return model

def get_optim(config, model):
    if config.optimizer =='Adam':
        optimizer = optim.Adam(model.parameters(), lr = float(config.lr))
    if config.optimizer =='RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr = float(config.lr))
    if config.optimizer =='Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr = float(config.lr))
    if config.optimizer =='SGD':
        optimizer = optim.SGD(model.parameters(), lr = float(config.lr))
    if config.optimizer =='Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr = float(config.lr))
    if config.optimizer =='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr = float(config.lr))
    if config.optimizer =='SparseAdam':
        optimizer = optim.SparseAdam(model.parameters(), lr = float(config.lr))
    if config.optimizer =='Adamax':
        optimizer = optim.Adamax(model.parameters(), lr = float(config.lr))
    if config.optimizer =='ASGD':
        optimizer = optim.ASGD(model.parameters(), lr = float(config.lr))
    if config.optimizer =='LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr = float(config.lr))
    if config.optimizer =='Rprop':
        optimizer = optim.Rprop(model.parameters(), lr = float(config.lr))
    print('\noptimizer :', optimizer,'\n')
    return optimizer

def config_device(config, model):
    parallel =False
    if config.cpu:
        device = torch.device('cpu')
    else:
        n_gpus = list(range(torch.cuda.device_count()))
        gpu_input = [num for num in config.gpu.replace(' ', '').split(',')]

        if len([i for i in gpu_input if int(i) in n_gpus]) == 0:
            print(gpu_input, 'is not available GPU')
            print('Available GPU :', n_gpus)
            exit()

        if len(gpu_input) == 1:
            device = torch.device("cuda:"+gpu_input[0])
            print('\nDevice :', device, '\n')
        else:
            device_list =[]
            for i in gpu_input:
                device_list.append(torch.device("cuda:"+i))
            model = nn.DataParallel(model, device_ids=device_list)
            device = device_list[0]
            parallel =True
            print('\nDevice :', device_list, '\n')
    model.to(device)

    return model, device, parallel