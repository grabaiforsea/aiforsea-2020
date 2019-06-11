from functools import reduce
from typing import Tuple, Optional, Mapping, Iterable, Any

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.models import Model


def collapse(layers: Iterable, initial_layer: Optional = None):
    return reduce(lambda first, second: second(first), layers, initial_layer)


def simple_model(image_size: Tuple[int, int], n_classes: int):
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

    output_layer = reduce(lambda first, second: second(first), layers, input_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile('sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def inception_v3_model(image_size: Tuple[int, int],
                       n_classes: int,
                       compile_kwargs: Optional[Mapping[str, Any]] = None):
    default_compile_kwargs = {'optimizer': 'sgd',
                              'loss': 'sparse_categorical_crossentropy',
                              'metrics': ['accuracy']}
    compile_kwargs = compile_kwargs or default_compile_kwargs

    inception_v3 = InceptionV3(include_top=False, pooling='max')

    top_layers = [Dense(2048, activation='relu'),
                  Dense(n_classes, activation='softmax')]
    top_combined = collapse(top_layers, inception_v3.output)

    model = Model(inputs=[inception_v3.input],
                  outputs=[top_combined])
    model.compile(**compile_kwargs)

    return model
