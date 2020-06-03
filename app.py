from gooey import Gooey, GooeyParser
import sys
import os
from subprocess import Popen, PIPE

REGRESSOR = ["Linear", "Ridge", "Lasso"]
CLASSIFIER = ["Decision Tree", "Ridge", "Lasso"]
CLUSTERING = ["K-means", "Label_Propagation"]

@Gooey(program_name="Tensorflow GUI")
def main():
    desc = "Choose your model"
    main_parser = GooeyParser(description=desc)
    model_sel_parser = main_parser.add_argument_group("Model Select", gooey_options={'show_border': True, 'columns': 1})

    model_kind = model_sel_parser.add_mutually_exclusive_group()
    model_kind.add_argument('--Regression',
                            choices=REGRESSOR,
                            dest = "Regression Model")

    model_kind.add_argument('--Classifier',
                            choices=CLASSIFIER,
                            dest="Classification Model")

    model_kind.add_argument('--Clustering',
                            choices=CLUSTERING,
                            dest="Clustering Model")

    args = main_parser.parse_args()
    for val in vars(args).values():
        if val is not None:
            model_name = val
            break;

    print("[Start]\t{}".format(model_name))
    PYTHON_PATH = sys.executable
    process = Popen([PYTHON_PATH, os.path.join('Model', model_name, 'run.py')], stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    # print(output)
    # print(error)

    print("[End]\t{}".format(model_name))


if __name__ == '__main__':
    main()
