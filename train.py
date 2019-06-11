import os

import numpy as np
import pandas as pd

from math import ceil

from scipy.io import loadmat

from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3



matlab_data = loadmat(os.path.join(os.curdir, 'devkit', 'cars_train_annos'))['annotations'][0, :]
annotations = np.stack([np.concatenate([array.ravel() for array in row])[4:] for row in matlab_data])
train_data = pd.DataFrame(annotations, columns=['target', 'filename'])
train_data['target'] = train_data['target']
train_data['filename'] = train_data['filename']

batch_size = 32
steps_per_epoch = ceil(len(train_data) * 0.8 / batch_size)
epochs = 150
validation_steps = ceil(len(train_data) * 0.2 / batch_size)

data_generator = ImageDataGenerator(validation_split=0.2)

train_iterator = data_generator.flow_from_dataframe(train_data, 
													directory='car_ims', 
													x_col='filename', 
													y_col='target', 
													class_mode='sparse', 
													batch_size=batch_size,
													seed=0,
													subset='training')
																
validation_iterator = data_generator.flow_from_dataframe(train_data, 
														 directory='car_ims', 
														 x_col='filename', 
														 y_col='target', 
														 class_mode='sparse', 
														 batch_size=batch_size,
														 seed=0,
														 subset='validation')

inception_v3 = InceptionV3(include_top=False, pooling='max')
model = Model(inputs=[inception_v3.input], outputs=[Dense(196, activation='softmax')(inception_v3.output)])
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_iterator, 
					steps_per_epoch=steps_per_epoch, 
					epochs=epochs,
					verbose=1, 
					validation_data=validation_iterator,
					validation_steps=validation_steps,
					max_queue_size=10)
					
model.save_weights('model.h5')