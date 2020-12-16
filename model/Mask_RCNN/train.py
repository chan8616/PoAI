from references.engine import train_one_epoch, evaluate
from datasets import *
from references import *
import torch
from build import *
from references import utils
import os


def do(config):
    # use our dataset and defined transformations
    if config.data_type == 'PennFudanPed':
        dataset = PennFudanDataset(config.root, get_transform(train=True))
        dataset_test = PennFudanDataset(config.root, get_transform(train=False))
    else:
        dataset = UserDataset(config.root, get_transform(train=True))
        dataset_test = UserDataset(config.root, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config.batch_size,
        shuffle=False, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person

    # get the model using our helper function
    model = get_instance_segmentation_model(config.num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.step_size,
                                                   gamma=config.gamma)

    for epoch in range(config.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model, os.path.join(config.save_directory, config.ckpt_name))
