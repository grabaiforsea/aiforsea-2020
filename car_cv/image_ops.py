import os
from warnings import warn

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

_TRAIN_LENGTH = 6
_TEST_LENGTH = 5


def load_annotations(path: str) -> pd.DataFrame:
    """
    Loads bounding box and label information from a file representing a MATLAB matrix.

    Args:
        path: The path to the file containing MATLAB matrix data.

    Returns:
        A `DataFrame` with columns representing the bounding box and label information. If the input data can be parsed
        as a matrix with 6 columns, those columns will be assumed to be, in order, `x_min`, `y_min`, `x_max`, `y_max`,
        `target` and `file_name`. If the input can be parsed as a matrix with 5 columns, then the `target` column will
        be assumed to have been omitted.
    """
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
    """
    Crops an image, given bounding box information.

    Args:
        image_data: The array representing an image.
        bb_x_min: The left edge of the bounding box.
        bb_x_max: The right edge of the bounding box.
        bb_y_min: The bottom edge of the bounding box.
        bb_y_max: The top edge of the bounding box.
        pad: The number of pixels to add to all edges of the bounding box.

    Returns:
        An array representing the cropped image.

    Notes:
        Each row (first dimension) of the input array is assumed to represent a row of pixels, and likewise for
        columns. Accordingly, the value of a pixel at `(x, y)` in the image represented by an array `a` would be
        accessed by `a[y, x]`.
    """
    im_y_max, im_x_max = image_data.shape[:2]
    x_min, y_min = max(bb_x_min - pad, 0), max(bb_y_min - pad, 0)
    x_max, y_max = min(bb_x_max + pad, im_x_max), min(bb_y_max + pad, im_y_max)
    return image_data[y_min:y_max, x_min:x_max]


def check_dims(a: np.ndarray, warning_message: str):
    """
    Checks if the length of any axis in an array is 0.

    Args:
        a: The array to check.
        warning_message: The message to show as a `RuntimeWarning` if any axis has length 0.
    """
    if any(length == 0 for length in a.shape):
        warn(RuntimeWarning(warning_message))


def write_image(image_data: np.ndarray, output_dir: str, output_file_name: str):
    """
    Writes an array representing an image to disk, creating any necessary intermediate directories.

    Args:
        image_data: The array to write.
        output_dir: The name of the output directory.
        output_file_name: The name of the file, to be placed in `output_dir`, to write to..
    """
    try:
        os.makedirs(output_dir)

    except FileExistsError:
        pass

    output_path = os.path.join(output_dir, output_file_name)

    result = cv2.imwrite(output_path, image_data)
    if result is None:
        warn(RuntimeWarning(f'Failed to write image to {output_path}.'))
