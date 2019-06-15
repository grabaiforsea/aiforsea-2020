from functools import partial
from typing import Tuple, Optional, Mapping, Any, List

from keras.engine import Layer
from keras.layers import Conv2D, BatchNormalization, Activation

from car_cv.models.inception_v4 import _CHANNEL_AXIS


def bn_conv2d(filters: int,
              kernel_size: Tuple[int, int],
              strides: Tuple[int, int] = (1, 1),
              padding: str = 'valid',
              name: Optional[str] = None,
              conv2d_kwargs: Optional[Mapping[str, Any]] = None,
              bn_kwargs: Optional[Mapping[str, Any]] = None) -> List[Layer]:
    """
    Creates a 2D convolution layer followed by a batch normalisation layer. Inspired by:
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py

    Args:
        filters: The number of filters to pass to the Conv2D constructor.
        kernel_size: The kernel size to pass to the Conv2D constructor.
        strides: The stride width in each dimension to pass to the Conv2D constructor.
        padding: The padding mode to pass to the Conv2D constructor.
        name: The name to which "_conv2d" and "_bn" will be appended for the Conv2D and BatchNormalization layers
        respectively.
        conv2d_kwargs: Keyword arguments to pass to the Conv2D constructor.
        bn_kwargs: Keyword arguments to pass to the BatchNormalization constructor.

    Returns:
        A list containing the new layers.

    Notes:
        "bn_conv2d" stands for "batch-normalised 2D convolution", which is why the term "bn" comes first despite being
        the second layer.
    """
    conv2d_kwargs = conv2d_kwargs or {}
    bn_kwargs = bn_kwargs or {}

    if name is not None:
        conv2d_name = name + '_conv2d'
        bn_name = name + '_bn'

    else:
        conv2d_name = None
        bn_name = None

    layers = [Conv2D(filters, kernel_size, strides=strides, padding=padding, name=conv2d_name, **conv2d_kwargs),
              BatchNormalization(axis=_CHANNEL_AXIS, name=bn_name, scale=False, **bn_kwargs),
              Activation('relu')
              ]

    return layers


# Convenience aliases for common layer shapes.


bn_conv2d_1x1 = partial(bn_conv2d, kernel_size=(1, 1))
bn_conv2d_3x3 = partial(bn_conv2d, kernel_size=(3, 3))
