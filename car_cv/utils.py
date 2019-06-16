import os
import tarfile
from functools import reduce
from itertools import chain
from typing import Iterable, Tuple, Any

from sys import stdout

import requests
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


def stream_download(url: str, output_path: str):
    """
    Downloads a file in a streaming fashion (writing data to disk as it is received and then discarding it from
    memory).

    Args:
        url: The URL at which to find the file to be downloaded.
        output_path: The path where the downloaded file will be placed.
    """
    with requests.get(url, stream=True) as req:
        if req.status_code == 200:
            total_size = int(req.headers.get('content-length', None))
            downloaded_size = 0
            with open(output_path, 'wb') as f:
                last_output_length = 0
                for chunk in req.iter_content(chunk_size=4096):
                    downloaded_size += len(chunk)
                    if total_size:
                        output_string = (f'\rAmount downloaded: {downloaded_size:,}/{total_size:,} '
                                         f'({downloaded_size / total_size:.1%})').ljust(last_output_length)

                    else:
                        output_string = f'\rAmount downloaded: {downloaded_size:,}'.ljust(last_output_length)

                    last_output_length = len(output_string)
                    stdout.write(output_string)
                    stdout.flush()
                    f.write(chunk)

                stdout.write('\n')

        else:
            raise RuntimeError(f'Download failed; got HTTP code {req.status}')


def extract_tgz(archive_path: str, output_path: str):
    """
    Extracts files from a gzip-compressed tarball.

    Args:
        archive_path: The path to the archive.
        output_path: The path where the extracted files will be placed.
    """
    with tarfile.open(archive_path, 'r:gz') as f:
        f.extractall(output_path)


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
