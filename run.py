import os
from math import ceil

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

from car_cv.defaults import augmentation_kwargs, make_callbacks
from car_cv.models import instantiate_model
from car_cv.parsing import make_parser
from car_cv.utils import get_label_info

argument_spec_path = os.path.join('resources', 'run_argument_spec.json')

parser = make_parser(argument_spec_path)
args = parser.parse_args()

flow_kwargs = {'target_size': args.image_size,
               'class_mode' : 'sparse',
               'batch_size' : args.batch_size,
               'seed'       : args.seed}

if args.training_path:
    n_classes, n_samples = get_label_info(args.training_path)

if args.load_path is not None:
    model = load_model(args.load_path)

elif args.architecture is not None and args.training_path:
    # noinspection PyUnboundLocalVariable
    model = instantiate_model(args.architecture, (*args.image_size, 3), n_classes)

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

if args.prediction_path:
    prediction_data_generator = ImageDataGenerator()
    prediction_iterator = prediction_data_generator.flow_from_directory(args.prediction_path, **flow_kwargs)
    result = model.predict_generator(prediction_iterator)
    result.save('prediction_result')
