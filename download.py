import os

from car_cv.parsing import make_parser
from car_cv.utils import stream_download, extract_tgz

argument_spec_path = os.path.join('resources', 'download_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

dataset_urls = {'train': 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz',
                'test': 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'}

devkit_urls = {'train': 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
               'test': 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'}

try:
    os.mkdir(args.output_folder)

except FileExistsError:
    pass

for dataset in args.datasets:
    dataset_path = os.path.join(args.output_folder, dataset)
    stream_download(dataset_urls[dataset], '_TEMPFILE')
    extract_tgz('_temp_dataset', dataset_path)
    os.remove('_TEMPFILE')

for devkit in args.devkit:
    devkit_path = os.path.join(args.output_folder, devkit)
    if devkit == 'train':
        stream_download(dataset_urls[devkit], '_TEMPFILE')
        extract_tgz('_temp_dataset', '_TEMPDIR')
        os.rename(os.path.join('_TEMPDIR', 'cars_train_annos.mat'), devkit_path)
        os.rmdir('_TEMPDIR')
        os.remove('_TEMPFILE')

    else:
        stream_download(dataset_urls[devkit], devkit_path)


