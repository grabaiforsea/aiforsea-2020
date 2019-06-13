import os

from car_cv.utils import load_annotations, process_images

annotations = load_annotations(os.path.join('devkit', 'cars_train_annos.mat'))

process_images(annotations, 'cars_train', os.path.join('output', 'cars_train'))
