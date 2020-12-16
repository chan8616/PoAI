
from tensorboard import program

def load_tfboard(config, save_dir):
    tfboard_path = save_dir + 'TensorBoard/'+ config.tensorboard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tfboard_path])
    tb.main()
    print('\nOpen Tensorboard :', config.tensorboard)