from model.keras_applications.build_config import BuildConfig, POOLINGS


class ResNet50Config(BuildConfig):
    NAME = 'ResNet50'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (224, 224, 3)  # type: ignore

    POOLING = POOLINGS[0]
