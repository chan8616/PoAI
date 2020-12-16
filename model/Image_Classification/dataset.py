import os
import sys
import numpy as np
import random
import torch
from torchvision import transforms
from torchvision import datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
import time
import glob

data_dir = "./Datasets/"
time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]

def make(config, save_dir):

    dataset_dir = save_dir + 'Dataset/'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    transform_list=[]
    if config.gray:
        transform_list.append(transforms.Grayscale())
    if config.resize != 0:
        transform_list.append(transforms.Resize((config.resize,config.resize)))
    if config.center_crop != 0:
        transform_list.append(transforms.CenterCrop(config.center_crop))
    if config.hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if config.vflip:
        transform_list.append(transforms.RandomVerticalFlip())
    if config.random_crop != 0:
        transform_list.append(transforms.RandomCrop(config.random_crop))
    if config.random_rot != 0:
        transform_list.append(transforms.RandomRotation(config.random_rot))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    user_dataset = vars(config)['Your Dataset']
    example_dataset = vars(config)['Prepared Dataset']
    if user_dataset is not None:
        if os.name=='nt':
            dataset_name = user_dataset.split('\\')[-1]
        else:
            dataset_name = user_dataset.split('/')[-1]
        print("Load your Dataset : ", dataset_name)

        list_d = np.array([os.path.isdir(os.path.join(user_dataset,f))
                            for f in os.listdir(user_dataset)])
        if np.all(list_d):
            dataset = datasets.ImageFolder(root = user_dataset, transform= transform)
        else:
            print("File and directory exist together!")
            sys.exit()

    else:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if example_dataset == 'MNIST':
            dataset = datasets.MNIST(root = data_dir, transform= transform, download=True)
        if example_dataset == 'FashionMNIST':
            dataset = datasets.FashionMNIST(root = data_dir, transform= transform, download=True)
        if example_dataset == 'CIFAR10':
            dataset = datasets.CIFAR10(root = data_dir, transform= transform, download=True)
        if example_dataset == 'CIFAR100':
            dataset = datasets.CIFAR100(root = data_dir, transform= transform, download=True)
        dataset_name = example_dataset

        print("Load {} Dataset\n".format(example_dataset))

    if config.save_sample:
        sample_data = [dataset[i][0].unsqueeze(0) for i in random.sample(range(len(dataset)), 64)]
        sample_data = torch.cat(sample_data)
        grid_img = vutils.make_grid(sample_data, padding=2, normalize=False).numpy()

        plt.figure(figsize=(8, 8))
        plt.title("Sample Data")
        plt.axis('off')

        plt.imshow(grid_img.transpose((1, 2, 0)))

        fig_name = dataset_dir + dataset_name +'_'+ time_stamp + '.jpg'
        plt.savefig(fig_name)
        print("Save sample data figure : ", fig_name)

    transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    transform = transforms.Compose(transform_list)
    dataset.transform = transform

    print('Transform List :')
    for t in transform_list:
        print(t)
    print('\n')

    dataset_dict = {}
    if config.split == 1:
        dataset_dict['train'] = dataset    
    else:
        len_train = int(len(dataset)*config.split)
        len_valid = int(len_train*config.valid_split)
        len_train -= len_valid

        idx = list(range(len(dataset)))
        train_idx = idx[:len_train]
        valid_idx = idx[len_train:len_train + len_valid]
        test_idx = idx[len_train+len_valid:]

        if len(train_idx) > 0:
            dataset_dict['train'] = Subset(dataset, train_idx)
        if len(valid_idx) > 0:
            dataset_dict['valid'] = Subset(dataset, valid_idx)
        if len(test_idx) > 0:
            dataset_dict['test'] = Subset(dataset, test_idx)

    for key, dataset in dataset_dict.items():
        save_path = dataset_dir + dataset_name +'_{}_'.format(key)+ time_stamp + '.dataset'
        with open(save_path, 'wb') as f:
            print("Save Dataset : ", save_path)
            torch.save(dataset, f)
