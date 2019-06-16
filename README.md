## Overview

This is my submission for the **Computer Vision** Grab AI for SEA Challenge, consisting of:

* A trained model
* Python scripts to perform:
	* Downloading of datasets and devkits
	* Image cropping and directory organisation	
	* Training, evaluation and prediction
* A small library containing utilities supporting the scripts
* A notebook to perform prediction

The model uses the Inception-ResNet V2 architecture (described in [this](http://arxiv.org/abs/1602.07261) paper), pretrained on imagenet data, and then trained on the training data from the [Stanford car dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). It has an accuracy of **89.47%** on the testing data from that dataset.

## Usage instructions

1. Install Python >= 3.6.
2. Clone this repository.
3. In the repository's root folder, run `pip install -r requirements.txt`. If you have all the packages specified in `requirements.txt` installed, skip this step.
4. Run `python download.py --datasets train test --devkits train test`. This will download the training and test datasets and devkits.
5. 
    1. Run `python preprocess.py download_output/devkit_train.mat download_output/cars_train/ --reorganise --crop`. This will crop the images and organise the results in a directory structure compatible with `keras`'s `ImageDataGenerator`.
    2. If you also wish to preprocess the test images in this way, run `python preprocess.py download_output/devkit_test.mat download_output/cars_test/ --reorganise --crop`.
6.  
    1. If you wish to train a model from scratch, run `python run.py --train preprocess_output/cars_train`. To perform testing on the test dataset, you can add `--evaluate preprocess_output/cars_test` to the same command. Not specifying `--architecture` will lead to automatically using the Inception-ResNet V2 architecture.
    2. Otherwise, download the [trained Inception-ResNet V2 model](https://github.com/marcuslimdw/aiforsea-2019/releases/download/v1.0/aiforsea-model-20190615-150602-30-0.4555.h5). To evaluate the model, run `python run.py --evaluate preprocess_output/cars_test --load-model aiforsea-model-20190615-150602-30-0.4555.h5`.
7. To perform prediction (inference) on an unseen dataset using the trained model, run `python run.py --predict <path_to_dataset> <path_to_output> --load_model aiforsea-model-20190615-150602-30-0.4555.h5`. For convenience, a Jupyter notebook doing the same thing is available as `prediction_notebook.ipynb`.

## Notes

Imports in the scripts do not follow PEP8 guidelines. This is because importing `keras` and `tensorflow` can take a while, so it makes sense to defer importing them to after the command-line arguments have been minimally checked for validity.

Since this is, more or less, a solved problem, a large part of my time went into wrapping the modelling process in an application layer. Due to the nature of this problem, the application is still highly specific, where possible I have endeavoured to write modular, extensible and well-documented code (all externally visible functions have docstrings and type annotations).

I briefly considered writing a simple test suite for this project, but rejected the idea because it would not be a good use of my time, and there really would not be much that could be robustly and easily tested, given the extremely high level of abstraction we work at ([this](https://ai.google/research/pubs/pub43146) paper seems relevant).

One thing I experimented with was creating my own implementation of Inception V4, which can be found in `/car_cv/models/`. An improvement over existing implementations that I have seen is in the way it is defined: a *specification* can be created using Python builtin collections (`lists` and `tuples`), which can exhaustively describe an architecture. This specification can then be recursively parsed to generate a model, applying the *functional programming* paradigm to `keras`'s "functional API". An example comparison:

```python
# with the base functional API
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

i = Input(shape=(299, 299, 3))
x = Conv2D(32, kernel_size=(3, 3))(i)
x = Activation('relu')(x)
x = Conv2D(32, kernel_size=(3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dense(50)(x)
x = Activation('softmax')(x)

normal_model = Model(inputs=[i], outputs=[x])

# with recursive parsing of a specification

from car_cv.models.model_builder import build_from_spec

input_tensor = Input(shape=(299, 299, 3))
layers = [Conv2D(32, kernel_size=(3, 3)),
		  Activation('relu'),
		  Conv2D(32, kernel_size=(3, 3)),
		  Activation('relu'),
		  Conv2D(32, kernel_size=(3, 3)),
		  MaxPooling2D(),
		  Dropout(0.2),
		  Flatten(),
		  Dense(512),
		  Activation('relu'),
		  Dense(50),
		  Activation('softmax')
		  ]

output_tensor = build_from_spec(layers, input_tensor)
spec_model = Model(inputs=[input_tensor], outputs=[output_tensor])
```

While they achieve the same *result*, I believe that not repeatedly applying layers to a dummy variable `x` is a cleaner way of doing so. In addition, there is space in a future project to serialise and deserialise this representation as a human-readable way of representing a model's architecture.

I was unable to perform pre-training of my Inception V4 implementation, and directly training the model on the Stanford car dataset led to a minimal decrease in loss per epoch. Given the limited time and resources, and the satisfactory performance on the pre-trained model, I decided not to explore this avenue any further
