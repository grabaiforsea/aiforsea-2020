import os
from functools import reduce
from itertools import chain
from typing import Iterable, Optional, Tuple

import requests


def collapse(layers: Iterable, initial_layer: Optional = None):
    return reduce(lambda first, second: second(first), layers, initial_layer)


def len_iter(iterable: Iterable) -> int:
    return sum(1 for _ in iterable)


def get_label_info(path: str) -> Tuple[int, int]:
    all_dirs, all_files = zip(*((dir_names, file_names) for _, dir_names, file_names in os.walk(path)))
    all_dirs = list(all_dirs)
    n_dirs = len_iter(chain.from_iterable(all_dirs))
    n_files = len_iter(chain.from_iterable(all_files))
    return n_dirs, n_files


def stream_download(url: str, output_path: str):
    with requests.get(url, stream=True) as req:
        if req.status == 200:
            with open(output_path, 'wb') as f:
                for chunk in req.iter_content(16384):
                    f.write(chunk)

        else:
            raise RuntimeError(f'Download failed; got HTTP code {req.status}')
