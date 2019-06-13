import os

from car_cv.parsing import make_parser
from car_cv.utils import stream_download

argument_spec_path = os.path.join('resources', 'download_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

dataset_urls = {'train': '',
                'test': ''}

devkit_urls = {'train': '',
               'test': ''}

for dataset in args.datasets:
    stream_download(dataset_urls[dataset], f'{os.path.join(args.output_folder, dataset)}')

for devkit in args.devkit:
    stream_download(dataset_urls[devkit], f'{os.path.join(args.output_folder, devkit)}')
