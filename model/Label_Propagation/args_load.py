from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser(), model_savefiles=[]):

    parser.add_argument('load_model',
                        metavar='Load Model',
                        widget='Dropdown',
                        choices= model_savefiles,
                        gooey_options={
                            'validator': {
                                'test': 'user_input != "Select Option"',
                                'message': 'Choose a save file from the list'}})

