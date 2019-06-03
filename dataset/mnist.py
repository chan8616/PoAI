import os
from pathlib import Path
import pandas as pd
import numpy as np

from keras.datasets import mnist as data
from keras.preprocessing.image import ImageDataGenerator

DEFAULT_DIR = 'dataset'
SAVE_PREFIX = Path(data.__file__).stem
DEFAULT_SAVE_DIR = os.path.join(DEFAULT_DIR, SAVE_PREFIX)

for i, (x, y) in enumerate(data.load_data()):
    x = np.expand_dims(x, -1)
    save_dir = os.path.join(DEFAULT_SAVE_DIR, 'test') if i else \
        os.path.join(DEFAULT_SAVE_DIR, 'train')

    if Path(save_dir).exists():
        for f in Path(save_dir).glob('*'):
            f.unlink()
        Path(save_dir).rmdir()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for _, _ in ImageDataGenerator().flow(x,
                                          y,
                                          save_to_dir=save_dir,
                                          save_prefix=SAVE_PREFIX,
                                          shuffle=False,
                                          batch_size=len(x),
                                          ):
        break

    paths = sorted([f.as_posix().replace(DEFAULT_SAVE_DIR+"/", "") for f in
                    Path(save_dir).glob('*.png')],
                   key=(lambda p: int(p.split('_')[1])))
    dataframe = pd.DataFrame(np.stack((paths, y), axis=1))
    pd.DataFrame(dataframe).to_csv(save_dir+'.csv',
                                   header=['path', 'label'],
                                   index=False)
    """
    dataframe_ = pd.read_csv(save_dir+'.csv', dtype=str)
    print(dataframe_)

    from matplotlib import pyplot as plt
    for xs, ys in ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=dataframe_,
        directory=DEFAULT_SAVE_DIR,
        batch_size=10,
        x_col='path',
        y_col='label',
        target_size=(32, 32),
    ):

        for x, y in zip(xs, ys):
            plt.figure()
            plt.imshow(x)
            plt.title(str(np.argmax(y)))
            plt.show()
        break
    """
