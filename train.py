import os

from functools import reduce
from datetime import datetime

import numpy as np
import pandas as pd

from math import ceil

from scipy.io import loadmat

from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

matlab_data = loadmat(os.path.join(os.curdir, 'devkit', 'cars_train_annos'))['annotations'][0, :]
annotations = np.stack([np.concatenate([array.ravel() for array in row])[4:] for row in matlab_data])
train_data = pd.DataFrame(annotations, columns=['target', 'filename'])

n_classes = train_data['target'].nunique()

batch_size = 32
steps_per_epoch = ceil(len(train_data) * 0.8 / batch_size)
validation_steps = ceil(len(train_data) * 0.2 / batch_size)
epochs = 150

data_generator = ImageDataGenerator(rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    brightness_range=(0.8, 1.2),
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    validation_split=0.2)

flow_kwargs = {'directory' : 'car_ims',
               'x_col'     : 'filename',
               'y_col'     : 'target',
               'class_mode': 'sparse',
               'batch_size': batch_size,
               'seed'      : 0}

train_iterator = data_generator.flow_from_dataframe(train_data,
                                                    **flow_kwargs,
                                                    subset='training')

validation_iterator = data_generator.flow_from_dataframe(train_data,
                                                               **flow_kwargs,
                                                               subset='validation')

callbacks = [ModelCheckpoint('aiforsea-model-{epoch}-{val_loss:.4f}.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'),
             ReduceLROnPlateau(factor=0.2,
                               patience=3),
             CSVLogger(f"aiforsea-model-{datetime.now().strftime('%Y%m%d-%H%m%S')}.csv")]
			 
inception_v3 = InceptionV3(include_top=False, pooling='max')

top_layers = [Dense(n_classes, activation='softmax')]
top_combined = reduce(lambda first, second: second(first), top_layers, inception_v3.output)

model = Model(inputs=[inception_v3.input],
              outputs=[top_combined])

model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_iterator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_iterator,
                    validation_steps=validation_steps,
                    max_queue_size=10)
