import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time
import os
import numpy as np
import glob
from build_model import config_device, get_model
import pandas as pd
from utils import make_savedir

time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
if os.name=='nt':
    sep = '\\'
else:
    sep = '/'
class PredictDataset(Dataset):
    def __init__(self, root, transform=None):
        self.name = root.split(sep)[-1]
        self.imgs=[]
        for ext in datasets.folder.IMG_EXTENSIONS:
            self.imgs += glob.glob(root + '/*' + ext)
        self.loader = datasets.folder.pil_loader
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        file_name = path.split(sep)[-1]
        return file_name, sample

    def __len__(self):
        return len(self.imgs)

def predict(config, save_dir):
    result_dir = save_dir + 'Result/'
    make_savedir(result_dir)


    #load model
    model_name, in_ch, num_classes, width, _, _, _, _ = config.modelLoad.split('_')
    model = get_model(model_name, int(in_ch), int(num_classes), False)
    # model.load_state_dict(torch.load(save_dir+'Model/' + config.modelLoad))
    model = torch.load(save_dir+'Model/' + config.modelLoad)
    print("Load Model : ", config.modelLoad)
    classes = np.array(model.classes)
    model, device, parallel = config_device(config, model)

    #make dataset
    transform_list=[]
    transform_list.append(transforms.Resize((int(width),int(width))))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    transform = transforms.Compose(transform_list)

    dataset = PredictDataset(config.predict_dataset, transform)
    dataloader = DataLoader(dataset, batch_size= config.batch_size)
    print("Load your Dataset : ", dataset.name)

    start_time = time.time()
    print('Predict Start')
    print('-' * 60)

    model.eval()
    list_file_names=[]
    list_pred=[]
    list_confidence = []

    with torch.no_grad():
        for file_name, inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            confidences, preds = torch.max(outputs, 1)

            list_file_names.append(file_name)
            list_pred.append(preds.cpu().detach().numpy())
            list_confidence.append(confidences.cpu().detach().numpy())
    
    arr_file_names = np.array(list_file_names).reshape(-1)
    arr_pred = classes[np.array(list_pred).reshape(-1)]
    arr_confidence = np.array(list_confidence).reshape(-1)

    result_df = pd.DataFrame({'file_names': arr_file_names,
                                "prediction" : arr_pred,
                                "confidence": arr_confidence})

    save_path = result_dir + dataset.name +'_'+ time_stamp + '.csv'
    print("save result :", save_path)
    result_df.to_csv(save_path)
    
    time_elapsed = time.time()-start_time
    print('elapsed time : {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed%60))
    print('-' * 60)
    print('\nComplete Predict\n')