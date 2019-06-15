from typing import Tuple, Mapping, Any, Iterable, NamedTuple

from keras import applications
from keras.engine import Layer
from keras.layers import Dense, Flatten
from keras.models import Model

from car_cv.defaults import compile_kwargs as default_compile_kwargs
from car_cv.utils import collapse


class PrebuiltArchitecture(NamedTuple):
    base_model: Model
    top_layers: Iterable[Layer]
    pooling: str
    compile_kwargs: Mapping[str, Any]


prebuilts = {'inception_resnet_v2': PrebuiltArchitecture(base_model=applications.InceptionResNetV2,
                                                         top_layers=[],
                                                         pooling='avg',
                                                         compile_kwargs=default_compile_kwargs),
             'inception_v3': PrebuiltArchitecture(base_model=applications.InceptionV3,
                                                  top_layers=[],
                                                  pooling='avg',
                                                  compile_kwargs=default_compile_kwargs),
             'VGG19': PrebuiltArchitecture(base_model=applications.VGG19,
                                           top_layers=[Flatten(),
                                                       Dense(4096),
                                                       Dense(4096)],
                                           pooling='max',
                                           compile_kwargs=default_compile_kwargs)
             }


def instantiate_prebuilt(architecture: PrebuiltArchitecture,
                         image_size: Tuple[int, int],
                         n_classes: int) -> Model:
    """
    Instantiates a `keras.Model` for image classification from a list of predefined architectures, using weights
    trained on imagenet data.

    Args:
        architecture: The object describing the architecture.
        image_size: The (x, y) dimensions of the images to classify.
        n_classes: The number of classes on which to perform classification.

    Returns:
        A `keras.Model` based on the given architecture..
    """
    base_model = architecture.base_model(include_top=False,
                                         weights='imagenet',
                                         input_shape=image_size,
                                         pooling=architecture.pooling)
    classification_layer = Dense(n_classes, activation='softmax')
    output_tensor = classification_layer(collapse(architecture.top_layers, base_model.output))
    model = Model(inputs=[base_model.input],
                  outputs=[output_tensor])
    model.compile(**architecture.compile_kwargs)

    return model
