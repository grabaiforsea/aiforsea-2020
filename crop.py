import os
from warnings import warn

import cv2
import pandas as pd

from car_cv.parsing import make_parser
from car_cv.utils import check_dims, write_image, crop_image

argument_spec_path = os.path.join('resources', 'crop_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

image_path = args.image_path

bb_info = pd.read_csv(args.bb_info_path)[['x_min', 'y_min', 'x_max', 'y_max', 'file_name']]

for _, (bb_x_min, bb_y_min, bb_x_max, bb_y_max, file_name) in bb_info.iterrows():
    input_path = os.path.join(image_path, file_name)
    output_dir = os.path.join(args.output_path, image_path)

    image = cv2.imread(input_path)
    if image is None:
        warn(RuntimeWarning(f'Failed to load image from {input_path}.'))

    cropped = crop_image(image, bb_x_min, bb_x_max, bb_y_min, bb_y_max)
    check_dims(cropped, f'The cropped image from {input_path} has at least one 0-length axis.')

    write_image(cropped, output_dir, file_name)
