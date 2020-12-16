import os
from tensorboard import program

def make_savedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def model_savefiles(save_dir):
    make_savedir(save_dir+'Model/')
    return list(sorted([save_file
                        for save_file in os.listdir(save_dir+'Model/')
                        if '.pth' in save_file], reverse=True))

def tensorboard_savefiles(save_dir):
    make_savedir(save_dir+'Tensorboard/')
    return list(sorted([save_file
                        for save_file in os.listdir(save_dir+'Tensorboard/')
                        if os.path.isdir(save_dir+'Tensorboard/'+save_file)], reverse=True))

def dataset_files(save_dir, dset):
    make_savedir(save_dir+'Dataset/')
    return list(sorted([save_file
                        for save_file in os.listdir(save_dir+'Dataset/')
                        if ('.dataset' in save_file) and (dset in save_file)], reverse=True))

def load_tfboard(config, save_dir):
    tfboard_path = save_dir + 'TensorBoard/'+ config.tensorboard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tfboard_path])
    tb.main()
    print('\nOpen Tensorboard :', config.tensorboard)