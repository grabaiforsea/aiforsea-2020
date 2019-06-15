from typing import Tuple

from keras import Input, Model
from keras.layers import Dense, AveragePooling2D, MaxPooling2D, Flatten, Dropout

from car_cv.models.layers import bn_conv2d_3x3, bn_conv2d_1x1, bn_conv2d
from car_cv.models.model_builder import build_from_spec


# This module is my implementation of InceptionV4, based on the https://arxiv.org/pdf/1602.07261v1.pdf, with some
# differences from the architecture laid out in the paper's diagrams.
#
# Stem:
# - The last Conv2D layer should have strides (2, 2) to match the other branch's MaxPooling2D layer's output.
# - The last MaxPooling2D should have pool size (3, 3) to match the others (referring also to keras's InceptionResNetV2
#   implementation)
#
# Block A:
# - The AveragePooling2D layer should have strides (1, 1) to match the other branches' outputs.
#
# Block B:
# - The last Conv2D layer in the third branch should probably have kernel shape (7x1) to match the previous layer, if
#   it is to be its counterpart in an effective 7x7 convolution.
#

# A note: I believe there is actually no need to specify padding='same' for the 1x1 convolutional layers, but I have
# left it in in the interests of consistency.


def inception_v4(input_shape: Tuple[int, int], n_classes: int, channel_axis: int) -> Model:

    input_tensor = Input((*input_shape, 3))
    classification_layer = Dense(n_classes, activation='softmax')

    model_base = build_from_spec(_COMBINED_SPEC, input_tensor, channel_axis)
    output_tensor = classification_layer(model_base)

    return Model(inputs=[input_tensor], outputs=[output_tensor])

_A_COUNT = 4
_B_COUNT = 7
_C_COUNT = 3

_STEM_SPEC = [[bn_conv2d_3x3(32, strides=(2, 2)),
               bn_conv2d_3x3(32),
               bn_conv2d_3x3(64, padding='same')
              ],
              (MaxPooling2D((3, 3), strides=(2, 2)),
               bn_conv2d_3x3(96, strides=(2, 2))
               ),
              ([bn_conv2d_1x1(64, padding='same'),
                bn_conv2d_3x3(96)
                ],
               [bn_conv2d_1x1(64, padding='same'),
                bn_conv2d(64, kernel_size=(7, 1), padding='same'),
                bn_conv2d(64, kernel_size=(1, 7), padding='same'),
                bn_conv2d_3x3(96)
                ]
               ),
              (bn_conv2d_3x3(192, strides=(2, 2)),
               MaxPooling2D((3, 3), strides=(2, 2)))
              ]

_A_BLOCK_SPEC = [([AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
                   bn_conv2d_1x1(96, padding='same')
                   ],
                  bn_conv2d_1x1(96, kernel_size=(1, 1), padding='same'),
                  [bn_conv2d_1x1(64, padding='same'),
                   bn_conv2d_3x3(96, padding='same'),
                   ],
                  [bn_conv2d_1x1(64, padding='same'),
                   bn_conv2d_3x3(96, padding='same'),
                   bn_conv2d_3x3(96, padding='same')]
                  )
                 ]

_A_REDUCTION_SPEC = [(MaxPooling2D((3, 3), strides=(2, 2)),
                      bn_conv2d_3x3(384, strides=(2, 2)),
                      [bn_conv2d_1x1(192, padding='same'),
                       bn_conv2d_3x3(224, padding='same'),
                       bn_conv2d_3x3(256, strides=(2, 2))
                       ]
                      )
                     ]

_B_BLOCK_SPEC = [([AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
                   bn_conv2d_1x1(128, padding='same')
                   ],
                  bn_conv2d_1x1(384),
                  [bn_conv2d_1x1(192, padding='same'),
                   bn_conv2d(224, kernel_size=(1, 7), padding='same'),
                   bn_conv2d(256, kernel_size=(1, 7), padding='same')
                  ],
                  [bn_conv2d_1x1(192, padding='same'),
                   bn_conv2d(192, kernel_size=(1, 7), padding='same'),
                   bn_conv2d(224, kernel_size=(7, 1), padding='same'),
                   bn_conv2d(224, kernel_size=(1, 7), padding='same'),
                   bn_conv2d(256, kernel_size=(7, 1), padding='same'),
                   ]
                  )
                 ]

_B_REDUCTION_SPEC = [(MaxPooling2D((3, 3), strides=(2, 2)),
                      [bn_conv2d_1x1(192, padding='same'),
                       bn_conv2d_3x3(192, strides=(2, 2))
                       ],
                      [bn_conv2d_1x1(256, padding='same'),
                       bn_conv2d(256, kernel_size=(1, 7), padding='same'),
                       bn_conv2d(320, kernel_size=(7, 1), padding='same'),
                       bn_conv2d_3x3(320, strides=(2, 2))
                       ]
                      )
                     ]

_C_BLOCK_SPEC = [([AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
                   bn_conv2d_1x1(256, padding='same')
                   ],
                  bn_conv2d_1x1(256, padding='same'),
                  [bn_conv2d_1x1(384, padding='same'),
                   (bn_conv2d(256, kernel_size=(3, 1), padding='same'),
                    bn_conv2d(256, kernel_size=(1, 3), padding='same')
                    )
                   ],
                  [bn_conv2d_1x1(384, padding='same'),
                   bn_conv2d(448, kernel_size=(1, 3), padding='same'),
                   bn_conv2d(512, kernel_size=(3, 1), padding='same'),
                   (bn_conv2d(256, kernel_size=(3, 1), padding='same'),
                    bn_conv2d(256, kernel_size=(1, 3), padding='same')
                    )
                   ]
                  )
                 ]

_TOP_SPEC = [AveragePooling2D((8, 8)),
             Dropout(0.2),
             Flatten()
             ]

_COMBINED_SPEC = [_STEM_SPEC,
 _A_BLOCK_SPEC * _A_COUNT,
 _A_REDUCTION_SPEC,
 _B_BLOCK_SPEC * _B_COUNT,
 _B_REDUCTION_SPEC,
 _C_BLOCK_SPEC * _C_COUNT,
 _TOP_SPEC]
