import os
from warnings import warn

from car_cv.parsing import make_parser

argument_spec_path = os.path.join('resources', 'preprocess_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()


import cv2

from car_cv.image_ops import load_annotations, crop_image, check_dims, write_image

if args.reorganise or args.crop:
    image_path = args.image_path

    devkit_info = load_annotations(args.devkit_path)

    if args.reorganise:
        bb_info = devkit_info[['x_min', 'y_min', 'x_max', 'y_max', 'file_name', 'target']]

    else:
        bb_info = devkit_info[['x_min', 'y_min', 'x_max', 'y_max', 'file_name']]

    for _, (bb_x_min, bb_y_min, bb_x_max, bb_y_max, *target_and_file_name) in bb_info.iterrows():
        file_name = target_and_file_name[0]
        file_path = os.path.join(image_path, file_name)

        image = cv2.imread(file_path)
        if image is None:
            warn(RuntimeWarning(f'Failed to load image from {file_path}.'))

        if args.crop:
            result = crop_image(image, bb_x_min, bb_x_max, bb_y_min, bb_y_max)
            check_dims(result, (f'The cropped image from {file_path} has at least one 0-length axis. This could mean '
                                f'that the bounding boxes are incorrect.'))

        else:
            result = image

        if args.reorganise:
            path_suffix = os.path.basename(os.path.normpath(image_path))
            output_path = os.path.join(args.output_path, path_suffix, target_and_file_name[1])
            write_image(result, output_path, file_name)

        else:
            write_image(result, image_path, file_name)
