from functools import reduce
from typing import Tuple, List, Iterable, Union

from keras.layers import concatenate

from car_cv.utils import TensorType


def build_from_spec(spec: Iterable[Union[List, Tuple]], previous: TensorType, channel_axis: int = -1) -> TensorType:
    """
    Converts a specification, in the form of an iterable of keras layers, possibly including further `lists` and/or
    `tuples` thereof, into an output tensor. `lists` represent linearly connected layers; `tuples` should always contain
    `lists`, and represent the output of the preceding layer being piped to multiple inputs (of the initial layers in
    each  `list`). See the specification for InceptionV4 for an example.

    Args:
        spec: The model specification.
        previous: The input tensor.
        channel_axis: The index of the axis containing channels.

    Returns:
        An output tensor.
    """
    def reduction_func(tensor, obj):
        if isinstance(obj, (list, tuple)):
            return build_sub_block(tensor, obj)

        else:
            return obj(tensor)

    def collapse(layers, initial):
        return reduce(reduction_func, layers, initial)

    def build_sub_block(current, sub_spec):
        if isinstance(sub_spec, list):
            return collapse(sub_spec, current)

        elif isinstance(sub_spec, tuple):
            return concatenate([build_sub_block(current, branch_spec) for branch_spec in sub_spec], axis=channel_axis)

        else:
            return sub_spec(current)

    return reduce(build_sub_block, spec, previous)
