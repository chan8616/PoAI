from argparse import ArgumentParser, Action, ArgumentTypeError
from gooey import Gooey, GooeyParser

def eval_int(string):
    value = eval(string)
    if not isinstance(value, int):
        msg = "%r cannot be eval to int" % string
        raise ArgumentTypeError(msg)
    return value

@Gooey()
def main_parser(parser=GooeyParser()):
    parser = GooeyParser()
    parser.add_argument(
        '--foo', type=eval_int, default=3,
        gooey_options={
            'validator': {
                'test': "isinstance(eval(user_input), int)",
                'message': 'unvalid input number'
            }
        })
    return parser
parser = main_parser()
args = parser.parse_args()
print(args)
