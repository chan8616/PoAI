import os
from pathlib import Path

from keras.datasets import cifar10 as data
from keras.preprocessing.image import ImageDataGenerator

DEFAULT_DIR = 'dataset'
SAVE_PREFIX = Path(data.__file__).stem
DEFAULT_SAVE_DIR = os.path.join(DEFAULT_DIR, SAVE_PREFIX)
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

for i, (x, y) in enumerate(data.load_data()):
    save_dir = os.path.join(DEFAULT_SAVE_DIR, 'test') if i else \
        os.path.join(DEFAULT_SAVE_DIR, 'train')

    for j, label in enumerate(LABELS):
        save_label_dir = os.path.join(save_dir, label)
        save_label_prefix = os.path.join(label, label)

        if Path(save_label_dir).exists():
            for f in Path(save_label_dir).glob('*'):
                f.unlink()
            Path(save_label_dir).rmdir()
        Path(save_label_dir).mkdir(parents=True, exist_ok=True)

        idx = (y == j).reshape(-1)
        for _, _ in ImageDataGenerator().flow(x[idx],
                                              y[idx],
                                              save_to_dir=save_label_dir,
                                              save_prefix=SAVE_PREFIX,
                                              shuffle=False,
                                              batch_size=sum(idx),
                                              ):
            print(sum(idx))
            break

    """
    import numpy as np
    from matplotlib import pyplot as plt
    for xs, ys in ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=save_dir,
        batch_size=10,
        target_size=(32, 32),
    ):

        for x, y in zip(xs, ys):
            plt.figure()
            plt.imshow(x)
            plt.title(LABELS[np.argmax(y)])
            plt.show()
        break
    """
