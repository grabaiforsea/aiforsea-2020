import os

from keras.models import load_model

from car_cv.models import instantiate_model
from car_cv.parsing import make_parser

argument_spec_path = os.path.join('resources', 'run_argument_spec.json')

parser, args = make_parser(argument_spec_path)

if args.train:
    pass


if args.load_path is None and args.architecture is None:
    parser.error('Either --load_path or --architecture must be specified.')

if args.load_path is not None:
    model = load_model(args.load_path)

else:
    model = instantiate_model(args.architecture, args.image_size)
