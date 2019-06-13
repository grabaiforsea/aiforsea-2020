import os

from car_cv.parsing import make_parser
from car_cv.utils import load_annotations, process_images

argument_spec_path = os.path.join('resources', 'prepare_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

# annotations = load_annotations(os.path.join('devkit', 'cars_train_annos.mat'))
#
# process_images(annotations, 'cars_train', os.path.join('output', 'cars_train'))

print(args)
