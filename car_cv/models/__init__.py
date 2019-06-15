from typing import Tuple

from keras import Model

from car_cv.defaults import compile_kwargs as default_compile_kwargs
from car_cv.models.inception_v4 import inception_v4

_MODELS = {'inception_v4': inception_v4}


def instantiate_custom(name: str,
                       image_size: Tuple[int, int],
                       n_classes: int) -> Model:
    try:
        model_constructor = _MODELS[name]

    except KeyError as e:
        raise KeyError(f'Could not find a model named {name}.') from e

    model = model_constructor(image_size, n_classes)
    model.compile(**default_compile_kwargs)

    return model
