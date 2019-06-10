import os

import numpy as np
import pandas as pd

from scipy.io import loadmat

from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

matlab_data = loadmat(os.path.join(os.curdir, 'devkit', 'cars_train_annos'))['annotations'][0, :]
annotations = np.stack([np.concatenate([array.ravel() for array in row])[4:] for row in matlab_data])
train_data = pd.DataFrame(annotations, columns=['target', 'filename'])
train_data['target'] = train_data['target']
train_data['filename'] = '0' + train_data['filename']
train_data_generator = ImageDataGenerator().flow_from_dataframe(train_data, directory='car_ims', x_col='filename', y_col='target', class_mode='sparse', seed=0)

inception_v3 = InceptionV3(include_top=False, pooling='max')
model = Model(inputs=inception_v3.input, outputs=Dense(196, activation='softmax')(inception_v3.output))
model.compile('adam', loss='sparse_categorical_crossentropy')

model.fit_generator(train_data_generator, steps_per_epoch=10)