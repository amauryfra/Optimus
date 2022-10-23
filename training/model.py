"""
This script holds the definition of the deep neural network model.
"""

# Imports
from keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import L2
from tensorflow import cast
from tensorflow import int8
from tensorflow import uint64
from tensorflow import float32
from tensorflow.math import floor
from tensorflow.math import abs as abs_tf
import sys
sys.path.insert(0, '..')
from utils import is_prime_tensor



def to_integer(x) :
    """
    This function transforms a real number into an odd integer and may be used 
    as the activation of the last layer of the neural network.

    Parameters
    ----------
    x : tensor - the last pre-activation

    Returns
    -------
    res : tensor - the odd integer output of the neural network

    """
    
    return 2 * cast(floor(abs_tf(x)), dtype = uint64) + 1



def to_float_integer(x) :
    """
    This function transforms a real number into an integer represented as a float
    (example : 10.0) and may be used as the activation of the last layer of the 
    neural network.

    Parameters
    ----------
    x : tensor - the last pre-activation

    Returns
    -------
    res : tensor - the floating point integer output of the neural network

    """
    
    return floor(abs_tf(x))



def to_binary(x) :
    """
    This function transforms a real number into a binary value 0 or 1 depending
    on the primality of its to_integer transformation. It may be used as the
    activation of the last layer of the neural network.

    Parameters
    ----------
    x : tensor - the last pre-activation

    Returns
    -------
    res : tensor - the binary output of the neural network -> 1 if output is prime

    """
    
    return cast(is_prime_tensor(to_integer(x)), dtype = int8)



def initialize_generator() :
    """
    This function initializes the deep neural network used for the prime number
    generation.

    Returns
    -------
    model : keras object - the deep neural network model

    """
    
    # Activation -> Leaky ReLU
    activ = LeakyReLU(alpha = 0.3)
    # L2 regularization to avoid having small outputs -> 
    # prime numbers are more concentrated within the small values
    l2_coeff = -1.0 * 1e6
    
    # Defining our neural network model
    # No bias to avoid approximating a constant function
    model = Sequential([
            InputLayer(input_shape = (1,)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(100, activation = activ, use_bias = False, kernel_initializer = 'he_normal', \
                  kernel_regularizer = L2(l2_coeff), activity_regularizer = L2(l2_coeff)),
            Dense(1, activation = abs_tf) # Output layer
        ])

    return model



def initialize_discriminator() :
    """
    This function initializes the deep neural network used for the prime number
    discrimination. It approximates the is_prime function.

    Returns
    -------
    model : keras object - the deep neural network model

    """
    
    # Activation -> ReLU + Sigmoid for probability
    # Usual regularization
    l2_coeff = 1e-3
    
    # Defining our neural network model
    model = Sequential([
            InputLayer(input_shape = (1,), dtype = float32),
            Dense(1000, activation = 'sigmoid', use_bias = True, kernel_initializer = 'glorot_normal', \
                  kernel_regularizer = L2(l2_coeff)),
            Dropout(0.3),
            Dense(100, activation = 'sigmoid', use_bias = True, kernel_initializer = 'glorot_normal', \
                  kernel_regularizer = L2(l2_coeff)),
            Dropout(0.3),
            Dense(100, activation = 'sigmoid', use_bias = True, kernel_initializer = 'glorot_normal', \
                  kernel_regularizer = L2(l2_coeff)),
            Dropout(0.3),
            Dense(100, activation = 'sigmoid', use_bias = True, kernel_initializer = 'glorot_normal', \
                  kernel_regularizer = L2(l2_coeff)),
            Dropout(0.3),
            Dense(100, activation = 'sigmoid', use_bias = True, kernel_initializer = 'glorot_normal', \
                  kernel_regularizer = L2(l2_coeff)),
            Dropout(0.3),
            Dense(100, activation = 'sigmoid', use_bias = True, kernel_initializer = 'glorot_normal', \
                  kernel_regularizer = L2(l2_coeff)),
            Dropout(0.3),
            Dense(1, activation = 'sigmoid') # Output layer
        ])
    
    return model


