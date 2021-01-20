import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score

from functools import partial


'''
classification
'''
def classifier(X, y, l2=0.001, dropout=1e-6, lr=0.006, 
               epochs=20, batch_size=32,seed=42):
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    model = keras.models.Sequential([layers.Masking(mask_value=0.0, 
                                                             input_shape=[None, X.shape[-1]]),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     layers.TimeDistributed(layers.Dense(15,activation='softmax'))
                                    ])

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    model.fit(X,y,epochs=epochs,
                  validation_split=0.2,batch_size=batch_size,verbose=1)
    return model

def encoder(X, l2=0.001, dropout=1e-6, lr=0.006,seed=42):
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    model = keras.models.Sequential([layers.Masking(mask_value=0.0, 
                                                             input_shape=[None, X.shape[-1]]),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     layers.TimeDistributed(layers.Dense(3,activation='linear')),
                                     layers.TimeDistributed(layers.Dense(15,activation='softmax'))
                                    ])

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    
    return model


'''
regression
'''
def regressor(l2=0, dropout=0, lr=0.001,seed=42):
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    model = keras.models.Sequential([layers.Masking(mask_value=0.0, 
                                                             input_shape=[None, 300]),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     CustomGRU(16,return_sequences=True),
                                     layers.TimeDistributed(layers.Dense(1,activation='linear'))
                                    ])
    # Optimizer
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='mse',
                      optimizer=optimizer)
    return model