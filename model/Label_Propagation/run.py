from gooey import Gooey, GooeyParser
import args_data, args_param, args_save
import train

@Gooey(optional_cols=2, program_name="Label Propagation", default_size=(800,600 ))
def run():
    parser = GooeyParser()
    subs = parser.add_subparsers(help='commands', dest='commands')

    train_parser = subs.add_parser('train', help='Configurate model training')
    param_group = train_parser.add_argument_group("Model parameter option", gooey_options={'show_border': True, 'columns': 2})
    param_group = args_param.add(param_group)
    data_group = train_parser.add_argument_group("Data Options", gooey_options={'show_border': True}, )
    data_group = args_data.add(data_group)
    log_group = train_parser.add_argument_group("Save option", gooey_options={'show_border': True, 'columns': 1})
    log_group = args_save.add(log_group)


    test_parser = subs.add_parser('test', help='Configurate model testining')
    data_group = test_parser.add_argument_group("Data Options", gooey_options={'show_border': True}, )
    data_group = args_data.add(data_group)
    log_group = test_parser.add_argument_group("Save option", gooey_options={'show_border': True, 'columns': 1})
    log_parser = args_save.add(log_group)


    args = parser.parse_args()

    if args.commands =='train':
        train.train(args)
    # else:
    #     test.test(args)

run()
