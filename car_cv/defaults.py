from datetime import datetime
from typing import List

from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.optimizers import SGD

n_channels = 3


augmentation_kwargs = {'rotation_range'  : 15,
                       'brightness_range': (0.9, 1.1),
                       'shear_range'     : 20,
                       'validation_split': 0.1}

compile_kwargs = {'optimizer': 'sgd',
                  'loss': 'sparse_categorical_crossentropy',
                  'metrics': ['accuracy']}

model_string = f"aiforsea-model-{datetime.now().strftime('%Y%m%d-%H%m%S')}" + '-{epoch}-{val_loss:.4f}'
csv_string = f"aiforsea-model-{datetime.now().strftime('%Y%m%d-%H%m%S')}.csv"


def make_callbacks(verbosity: int) -> List[Callback]:
    """
    Create a `list` of default callbacks to use for model training.

    Args:
        verbosity: The level of verbosity of the callbacks.

    Returns:
        A `list` of callbacks.
    """
    return [ModelCheckpoint(model_string,
                            monitor='val_loss',
                            verbose=verbosity,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='min',
                            period=3),
            ReduceLROnPlateau(factor=0.2,
                              verbose=verbosity,
                              patience=5,
                              min_delta=0.01),
            CSVLogger(csv_string)]
