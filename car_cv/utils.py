import os
from functools import reduce
from itertools import chain
from typing import Iterable, Tuple, Any

from keras.engine import Layer

TensorType = Any


def len_iter(iterable: Iterable) -> int:
    """
    Gets the length of an iterable.

    Args:
        iterable: The iterable to take the length of.

    Returns:
        The length of the iterable.
    """
    return sum(1 for _ in iterable)


def get_dir_info(path: str) -> Tuple[int, int]:
    """
    Counts the number of directories and files in a given directory, performing recursive descent into any
    subdirectories.

    Args:
        path: The path to the directory in which to search.

    Returns:
        n_dirs: The number of subdirectories in `path`, recursively calculated.
        n_files:  The number of files in `path`, recursively calculated.
    """
    all_dirs, all_files = zip(*((dir_names, file_names) for _, dir_names, file_names in os.walk(path)))
    n_dirs = len_iter(chain.from_iterable(all_dirs))
    n_files = len_iter(chain.from_iterable(all_files))
    return n_dirs, n_files


# This function returns a tensor, but the specific class may vary, depending on the backend used. Since the Tensor
# classes for each backend do not have a common supertype (except object, of course), the only way to properly annotate
# types for this function is with a union of all those classes. That is inefficient and rules out unseen backends, so
# we can just use a placeholder type here.
def collapse(layers: Iterable[Layer], initial: TensorType) -> TensorType:
    """
    Collapses an iterable of `keras.layers.Layer` into a single `Tensor`, using the "functional API".

    Args:
        layers: The iterable of layers to sequentially apply.
        initial: The initial tensor to apply layers to.

    Returns:
        A tensor representing the graph containing each layer as nodes.
    """
    return reduce(lambda first, second: second(first), layers, initial)
