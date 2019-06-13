import os
from warnings import warn

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

_TRAIN_LENGTH = 6
_TEST_LENGTH = 5


def load_annotations(path: str) -> pd.DataFrame:
    matlab_data = loadmat(path)['annotations'][0, :]
    annotations = np.stack([np.concatenate([array.ravel() for array in row]) for row in matlab_data])
    if annotations.shape[1] == _TRAIN_LENGTH:
        df = pd.DataFrame(annotations, columns=['x_min', 'y_min', 'x_max', 'y_max', 'target', 'file_name'])

    elif annotations.shape[1] == _TEST_LENGTH:
        df = pd.DataFrame(annotations, columns=['x_min', 'y_min', 'x_max', 'y_max', 'file_name'])

    else:
        raise ValueError('Unknown input data format.')

    df[['x_min', 'y_min', 'x_max', 'y_max']] = df[['x_min', 'y_min', 'x_max', 'y_max']].astype(int)
    return df


def crop_image(image_data: np.ndarray,
               bb_x_min: int,
               bb_x_max: int,
               bb_y_min: int,
               bb_y_max: int,
               pad: int = 16) -> np.ndarray:
    im_y_max, im_x_max = image_data.shape[:2]
    x_min, y_min = max(bb_x_min - pad, 0), max(bb_y_min - pad, 0)
    x_max, y_max = min(bb_x_max + pad, im_x_max), min(bb_y_max + pad, im_y_max)
    return image_data[y_min:y_max, x_min:x_max]


def check_dims(image_data: np.ndarray, warning_message: str):
    if any(length == 0 for length in image_data.shape):
        warn(RuntimeWarning(warning_message))


def write_image(image_data: np.ndarray, output_dir: str, output_file_name: str):
    try:
        os.makedirs(output_dir)

    except FileExistsError:
        pass

    output_path = os.path.join(output_dir, output_file_name)

    result = cv2.imwrite(output_path, image_data)
    if result is None:
        warn(RuntimeWarning(f'Failed to write image to {output_path}.'))
