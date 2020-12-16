from gooey import Gooey, GooeyParser
import os
import pickle
import pandas as pd
import json
import sys

import args_data, args_param, args_save, args_load
from train_test import train, test

save_dir = "./save_dir/Random_Forest/"

def make_savedir():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

def model_savefiles():
    return list(sorted([save_file
                        for save_file in os.listdir(save_dir)
                        if '.sav' in save_file], reverse=True))

def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1].values.reshape(-1)
    return X, Y

@Gooey(image_dir='image/ml',optional_cols=2,
       program_name="Random Forest",
       default_size=(600,800 ),
       poll_external_updates = True)
def run():
    parser = GooeyParser()
    subs = parser.add_subparsers(help='commands', dest='commands')

    train_parser = subs.add_parser('train', help='Configurate model training')
    param_group = train_parser.add_argument_group("Model parameter option", gooey_options={'show_border': True, 'columns': 2})
    args_param.add(param_group)
    data_group = train_parser.add_argument_group("Data Options", gooey_options={'show_border': True}, )
    args_data.add(data_group)
    save_group = train_parser.add_argument_group("Save option", gooey_options={'show_border': True, 'columns': 2})
    args_save.add(save_group)


    test_parser = subs.add_parser('test', help='Configurate model testining')
    data_group = test_parser.add_argument_group("Data Options", gooey_options={'show_border': True}, )
    args_data.add(data_group)
    load_group = test_parser.add_argument_group("Load option", gooey_options={'show_border': True, 'columns': 1})
    args_load.add(load_group, model_savefiles())
    save_group = test_parser.add_argument_group("Save option", gooey_options={'show_border': True, 'columns': 2})
    args_save.add(save_group)

    args = parser.parse_args()
    X, Y = load_data(args.data_path)

    if args.commands =='train':
        train(args, X, Y, save_dir)
    else:
        with open(save_dir + args.load_model, 'rb') as f:
            model = pickle.load(f)
        test(args, X, Y, save_dir, model)

make_savedir()
if 'gooey-seed-ui' in sys.argv:
    print(json.dumps({'load_model': model_savefiles()}))
else:
    run()
