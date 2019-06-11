import os
from datetime import datetime
from math import ceil

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from models import inception_v3_model

n_classes = 196

batch_size = 32
t = 8144
steps_per_epoch = ceil(t * 0.8 / batch_size)
epochs = 150
validation_steps = ceil(t * 0.2 / batch_size)

image_size = (224, 224)

data_generator = ImageDataGenerator(validation_split=0.2)

train_iterator = data_generator.flow_from_directory(os.path.join('output', 'cars_train'),
                                                    target_size=image_size,
                                                    class_mode='sparse',
                                                    batch_size=batch_size,
                                                    seed=0,
                                                    subset='training')

validation_iterator = data_generator.flow_from_directory(os.path.join('output', 'cars_train'),
                                                         target_size=image_size,
                                                         class_mode='sparse',
                                                         batch_size=batch_size,
                                                         seed=0,
                                                         subset='validation')

callbacks = [ModelCheckpoint('aiforsea-model-{epoch}-{val_loss:.4f}.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'),
             ReduceLROnPlateau(factor=0.2,
                               patience=3),
             CSVLogger(f"aiforsea-model-{datetime.now().strftime('%Y%m%d-%H%m%S')}.csv")]

model = inception_v3_model(image_size, n_classes)

model.fit_generator(generator=train_iterator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_iterator,
                    validation_steps=validation_steps,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1)
