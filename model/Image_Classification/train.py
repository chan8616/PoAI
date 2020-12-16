import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tensorboard import program
import time
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import webbrowser
from build_model import config_device, get_model, get_optim
from utils import make_savedir

time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]

def load_dataloader(config, save_dir):
    dataset={}
    len_dataset={}
    dataset_name = config.dataset_train.split('_')[0]
    print('\n')
    with open(save_dir + 'Dataset/' +config.dataset_train, 'rb') as f:
        dataset['train'] = torch.load(f).dataset
        len_dataset['train'] = len(dataset['train'])
    print("Load train dataset :",config.dataset_train)
    if hasattr(config, 'dataset_valid'):
        with open(save_dir +'Dataset/' + config.dataset_valid, 'rb') as f:
            dataset['valid'] = torch.load(f).dataset
            len_dataset['valid'] = len(dataset['valid'])
        print("Load valid dataset :",config.dataset_valid)
    if hasattr(config, 'dataset_test'):
        with open(save_dir + 'Dataset/' + config.dataset_test, 'rb') as f:
            dataset['test'] = torch.load(f).dataset
            len_dataset['test'] = len(dataset['test'])
        print("Load test dataset :", config.dataset_test)
    print('\n')

    if hasattr(config, 'shuffle'):
        shuffle = config.shuffle
    else:
        shuffle = False
    
    dataloader={}
    for key in dataset.keys():
        dataloader[key] = DataLoader(
            dataset[key],
            batch_size= config.batch_size,
            shuffle = shuffle)
    
    return dataloader, len_dataset , dataset_name

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img/2+0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

def images_to_probs(model, images):
    output = model(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model, images, classes, labels=None):
    preds, probs = images_to_probs(model, images)
    images = images.cpu()
    labels = labels.cpu()
    fig = plt.figure(figsize=(16, 16))
    one_channel = (images[0].shape[0] == 1)
    num_imgs = len(images)
    if num_imgs >16:
        num_imgs= 16
    for idx in np.arange(num_imgs):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=one_channel)

        if labels is not None:
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"), fontsize=16)
        else:
            ax.set_title("{0}, {1:.1f}%".format(classes[preds[idx]], probs[idx] * 100.0), fontsize=16)
    return fig

def get_tfboard_writer(save_dir, model_name, dataset_name, save_time = None):
    if save_time == None:
        save_time = time_stamp
    tfboard_path = save_dir + 'TensorBoard/'+'_'.join([model_name, dataset_name, save_time])
    make_savedir(tfboard_path)
    writer = SummaryWriter(tfboard_path)
    return writer, tfboard_path

def train_model(config, save_dir):

    model_path = save_dir + 'Model/'
    make_savedir(model_path)

    #load dataset
    dataloader, len_dataset, dataset_name = load_dataloader(config, save_dir)
    in_ch = dataloader['train'].dataset[0][0].shape[0]
    classes = dataloader['train'].dataset.classes
    width  = dataloader['train'].dataset[0][0].shape[-1]
    phases = dataloader.keys()
    num_classes = len(classes)

    #load model
    for key in vars(config).keys():
        if ('model' in key) and vars(config)[key] != None:
            model_name = vars(config)[key]
            break

    model = get_model(model_name, in_ch, num_classes, config.preTrain)
    model.classes = classes
    model, device, parallel = config_device(config, model)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optim(config, model)

    if config.save_best:
        best_model_wts = copy.deepcopy(model.state_dict())

    if config.tfboard:
        writer, tfboard_path = get_tfboard_writer(save_dir, model_name, dataset_name)
        images, _ = next(iter(dataloader['train'])) 
        if parallel:
            writer.add_graph(model.module, images.to(device))
        else:
            writer.add_graph(model, images.to(device))

    num_epochs = config.epoch
    best_acc = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('\n\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 60)

        for phase in phases:
            if phase == 'train':
                model.train()
            elif phase =='valid':
                model.eval()
            else:
                break
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader[phase]:
                if len(labels) == 1:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss = loss.item() *inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
            epoch_loss = running_loss / len_dataset[phase]
            epoch_acc = running_corrects.double() / len_dataset[phase]

            print('<{}>\t\tLoss : {:.4f}\t\tAcc : {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        
            if config.tfboard:
                writer.add_scalar(phase+'_loss', epoch_loss, global_step=epoch+1)
                writer.add_scalar(phase+'_acc', epoch_acc, global_step=epoch+1)

        if config.tfboard:
            writer.add_figure('valid : predictions vs. actuals',
                plot_classes_preds(model, inputs, classes, labels), global_step=epoch+1)
            if epoch == 0:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', tfboard_path])
                url = tb.launch()
                webbrowser.open(url)

        time_elapsed = time.time()-epoch_start_time
        print('elapsed time : {:.0f}m {:.0f}s'.format(
            time_elapsed//60, time_elapsed%60))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            if config.save_best:
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                path = model_path+'_'.join([model_name, str(in_ch), str(num_classes),dataset_name, '{:.4f}'.format(best_acc.item()),time_stamp]) + '.pth'
                print('save ', path.split('/')[-1])
                if parallel :
                    # torch.save(model.module.state_dict(), path)
                    torch.save(model.module, path)
                else:
                    # torch.save(model.state_dict(), path)
                    torch.save(model, path)
        print("Best valid Acc: {:4f}".format(best_acc))
        print('-' * 60)

    if config.save_best:
        model.load_state_dict(best_model_wts)
        path = model_path+'_'.join([model_name, str(in_ch), str(num_classes), str(width), dataset_name, '{:.4f}'.format(best_acc.item()),time_stamp]) + '.pth'
        print('save ', path)
        print('save ', path.split('/')[-1])
        if parallel :
            # torch.save(model.module.state_dict(), path)
            torch.save(model.module, path)
        else:
            # torch.save(model.state_dict(), path)
            torch.save(model, path)

    if 'test' in phases:
        test_model(model, dataloader['test'], classes, device)

    if config.tfboard:
        writer.close()
    print('\nComplete training\n')


def test_model(model, dataloader, classes, device):
    start_time = time.time()
    print('\n\nTest Start')
    print('-' * 60)
    class_correct = np.zeros(len(classes))
    class_total = np.zeros(len(classes))

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            c = (preds == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        test_acc = class_correct.sum() / class_total.sum()
        print('Test Acc : {:.4f}'.format(test_acc))
        for i in range(len(classes)):
            print("Accuracy of %10s : %.2f %% ( %d / %d )"%(classes[i], class_correct[i]/class_total[i]*100, class_correct[i], class_total[i]))
    time_elapsed = time.time()-start_time
    print('elapsed time : {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed%60))
    print('-' * 60)
