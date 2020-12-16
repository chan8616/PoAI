from gooey import Gooey, GooeyParser
import args_data, args_train, args_test
import train
import test

@Gooey(image_dir='image/nlp', optional_cols=2, program_name="Word2Vec", default_size=(800,600 ))
def run():
    parser = GooeyParser()
    subs = parser.add_subparsers(help='commands', dest='commands')

    train_parser = subs.add_parser('train', help='Training Configuration')
    param_group = train_parser.add_argument_group("Model", gooey_options={'show_border': True, 'columns': 2})
    param_group = args_train.add(param_group)
    data_group = train_parser.add_argument_group("Data", gooey_options={'show_border': True}, )
    data_group = args_data.add(data_group)

    test_parser = subs.add_parser('test', help='Test Configuration')
    log_group = test_parser.add_argument_group("Files", gooey_options={'show_border': True, 'columns': 1})
    log_parser = args_test.add(log_group)

    print(param_group)
    print(data_group)
    print(log_group)

    args = parser.parse_args()
    print(args)

    if args.commands == 'train':
        train.do(args)
    else:
        test.do(args)
run()
