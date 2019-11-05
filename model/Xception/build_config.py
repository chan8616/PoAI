from model.keras_applications.build_config import BuildConfig, POOLINGS


class XceptionConfig(BuildConfig):
    NAME = 'Xception'

    #  INPUT_SHAPE: Optional[Tuple]
    INPUT_SHAPE = (299, 299, 3)  # type: ignore

    POOLING = POOLINGS[1]
