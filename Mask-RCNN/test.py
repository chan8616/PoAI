from references.engine import train_one_epoch, evaluate
from datasets import *
from references import *
import torch
from build import *
from references import utils
import os
from PIL import Image


def do(config):
    # use our dataset and defined transformations
    if config.data_type == 'PennFudanPed':
        dataset = PennFudanDataset(config.root, get_transform(train=False))
    else:
        dataset = UserDataset(config.root, get_transform(train=False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=False, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torch.load(config.checkpoint_path)
    # move model to the right device
    model.to(device)

    # test model
    model.eval()

    with torch.no_grad():
        idx = 1
        for img, _ in data_loader:
            img = img[0]
            prediction = model([img.to(device)])
            img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            img_name = "img_"+str(idx)+'.jpg'
            img.save(os.path.join(config.save_directory, img_name))
            n_masks = list(prediction[0]['masks'].size())[0]
            print("number of detected objects in a image: ", n_masks)
            for i in range(n_masks):
                mask = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                mask_name = "mask_"+str(idx)+"_obj_"+str(i)+".jpg"
                mask.save(os.path.join(config.save_directory, mask_name))
            idx += 1