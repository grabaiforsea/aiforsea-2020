import os
import numpy as np
from math import ceil

from car_cv.parsing import make_parser

argument_spec_path = os.path.join('resources', 'run_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

from car_cv.defaults import augmentation_kwargs, make_callbacks, n_channels
from car_cv.models import instantiate_custom
from car_cv.prebuilt import prebuilts, instantiate_prebuilt
from car_cv.utils import get_dir_info


flow_kwargs = {'target_size': args.image_size,
               'class_mode' : 'sparse',
               'batch_size' : args.batch_size,
               'seed'       : args.seed}

if args.training_path:
    n_classes, n_samples = get_dir_info(args.training_path)

if args.load_path is not None:
    model = load_model(args.load_path)

elif args.architecture is not None and args.training_path:
    if args.architecture in prebuilts:
        # noinspection PyUnboundLocalVariable
        model = instantiate_prebuilt(prebuilts[args.architecture], (*args.image_size, n_channels), n_classes)

    else:
        # noinspection PyUnboundLocalVariable
        model = instantiate_custom(args.architecture, args.image_size, n_classes)

else:
    model = None
    parser.error('Either --load-path or --architecture and --train together must be specified.')

if args.training_path:
    training_data_generator = ImageDataGenerator(**augmentation_kwargs)
    training_iterator = training_data_generator.flow_from_directory(args.training_path,
                                                                    **flow_kwargs,
                                                                    subset='training')
    validation_iterator = training_data_generator.flow_from_directory(args.training_path,
                                                                      **flow_kwargs,
                                                                      subset='validation')

    # noinspection PyUnboundLocalVariable
    steps_per_epoch = ceil(n_samples * (1 - augmentation_kwargs['validation_split']) / args.batch_size)
    validation_steps = ceil(n_samples * augmentation_kwargs['validation_split'] / args.batch_size)

    model.fit_generator(training_iterator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs,
                        verbose=args.verbosity,
                        callbacks=make_callbacks(args.verbosity),
                        validation_data=validation_iterator,
                        validation_steps=validation_steps)
                        
    if args.save_path is not None:
        model.save(save_path)

if args.evaluation_path:
    evaluation_data_generator = ImageDataGenerator()
    evaluation_iterator = evaluation_data_generator.flow_from_directory(args.evaluation_path,
                                                                        shuffle=False,
                                                                        **flow_kwargs)
    evaluation_results = model.evaluate_generator(evaluation_iterator, verbose=args.verbosity)
    print('=' * 80)
    for metric, value in zip(model.metrics_names, evaluation_results):
        print(f'{metric}: {value}')

    print('=' * 80)


if args.prediction_paths:
    prediction_data_path, prediction_output_path = args.prediction_paths
    prediction_data_generator = ImageDataGenerator()
    prediction_iterator = prediction_data_generator.flow_from_directory(prediction_data_path,
                                                                        shuffle=False,
                                                                        **flow_kwargs)
    prediction_result = model.predict_generator(prediction_iterator, verbose=args.verbosity)
    np.save(prediction_output_path, prediction_result)
