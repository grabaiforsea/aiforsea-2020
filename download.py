import os
from shutil import rmtree

from car_cv.parsing import make_parser

argument_spec_path = os.path.join('resources', 'download_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

from car_cv.iolib import stream_download, extract_tgz

dataset_urls = {'train': 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz',
                'test': 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'}

devkit_urls = {'train': 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
               'test': 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'}

try:
    os.mkdir(args.output_path)

except FileExistsError:
    pass

for dataset_type in args.datasets:
    print(f'Downloading {dataset_type} dataset...')
    try:
        stream_download(dataset_urls[dataset_type], '_TEMPFILE')
        print(f'Done downloading {dataset_type} dataset.')
        extract_tgz('_TEMPFILE', args.output_path)
        print(f'{dataset_type} dataset successfully extracted to {args.output_path}')

    finally:
        os.remove('_TEMPFILE')


for devkit_type in args.devkits:
    print(f'Downloading {devkit_type} devkit...')
    devkit_path = os.path.join(args.output_path, f'devkit_{devkit_type}.mat')
    if devkit_type == 'train':
        try:
            stream_download(devkit_urls[devkit_type], '_TEMPFILE')
            extract_tgz('_TEMPFILE', '_TEMPDIR')
            os.rename(os.path.join('_TEMPDIR', 'devkit', 'cars_train_annos.mat'), devkit_path)

        finally:
            rmtree('_TEMPDIR')
            os.remove('_TEMPFILE')

    else:
        stream_download(devkit_urls[devkit_type], devkit_path)

    print(f'Done downloading {devkit_type} devkit.')
