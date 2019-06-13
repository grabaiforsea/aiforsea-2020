import os
from datetime import datetime
from math import ceil

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from car_cv.models import instantiate_model

n_classes = 196

batch_size = 10
steps_per_epoch = ceil(8144 * 0.8 / batch_size)
validation_steps = ceil(8144 * 0.2 / batch_size)
epochs = 100
test_steps = ceil(8041 / batch_size)

image_size = (299, 299)

train_data_generator = ImageDataGenerator(rotation_range=15,
                                          shear_range=0.1,
                                          validation_split=0.1)
test_data_generator = ImageDataGenerator()

train_iterator = train_data_generator.flow_from_directory(os.path.join('output', 'cars_train'),
                                                          target_size=image_size,
                                                          class_mode='sparse',
                                                          batch_size=batch_size,
                                                          seed=0,
                                                          subset='training')

validation_iterator = train_data_generator.flow_from_directory(os.path.join('output', 'cars_train'),
                                                               target_size=image_size,
                                                               class_mode='sparse',
                                                               batch_size=batch_size,
                                                               seed=0,
                                                               subset='validation')

test_iterator = test_data_generator.flow_from_directory(os.path.join('output', 'cars_test'),
                                                        target_size=image_size,
                                                        class_mode='sparse',
                                                        batch_size=batch_size,
                                                        seed=0)

callbacks = [
        ModelCheckpoint(f"aiforsea-model-{datetime.now().strftime('%Y%m%d-%H%m%S')}" + '-{epoch}-{val_loss:.4f}.h5',
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode='min',
                        period=3),
        ReduceLROnPlateau(factor=0.2,
                          verbose=1,
                          patience=4,
                          min_delta=0.01),
        CSVLogger(f"aiforsea-model-{datetime.now().strftime('%Y%m%d-%H%m%S')}.csv")]

model = instantiate_model('inception_resnet_v2', image_size, n_classes)

model.fit_generator(generator=train_iterator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_iterator,
                    validation_steps=validation_steps,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1)
