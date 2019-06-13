from typing import Tuple, Optional, Mapping, Any

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.models import Model

from car_cv.defaults import compile_kwargs as default_compile_kwargs
from car_cv.utils import collapse


def instantiate_model(architecture: str, image_size: Tuple[int, int], n_classes: int):
    """
    Instantiates a `keras.Model` for image classification from a list of predefined architectures.

    Args:
        architecture: The name of the architecture of the desired model.
        image_size: The (x, y) dimensions of the images to classify.
        n_classes: The number of classes on which to perform classification.

    Returns:
        A `keras.Model` based on the given architecture..
    """
    return _MODELS[architecture](image_size, n_classes)


def _simple(image_size: Tuple[int, int], n_classes: int):
    input_layer = Input(shape=(*image_size, 3))

    layers = [Conv2D(32, (3, 3), padding='same', activation='relu'),
              Conv2D(32, (3, 3), activation='relu'),
              MaxPooling2D(pool_size=(2, 2)),
              Dropout(0.25),
              Conv2D(32, (3, 3), padding='same', activation='relu'),
              Conv2D(32, (3, 3), activation='relu'),
              MaxPooling2D(pool_size=(2, 2)),
              Dropout(0.25),
              Flatten(),
              Dense(512, activation='relu'),
              Dense(n_classes, activation='softmax')]

    output_layer = collapse(layers, input_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile('sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def _inception_resnet_v2(image_size: Tuple[int, int],
                         n_classes: int,
                         compile_kwargs: Optional[Mapping[str, Any]] = None):
    compile_kwargs = compile_kwargs or default_compile_kwargs

    base_model = InceptionResNetV2(include_top=False, input_shape=image_size, pooling='avg')
    top_layers = [Dense(n_classes, activation='softmax')]
    top_combined = collapse(top_layers, base_model.output)

    model = Model(inputs=[base_model.input],
                  outputs=[top_combined])
    model.compile(**compile_kwargs)

    return model


_MODELS = {'simple': _simple,
           'inception_resnet_v2': _inception_resnet_v2}

