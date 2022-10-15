"""
This scripts holds the definition of the deep neural network model.
"""

# Imports
from keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.math import floor
from tensorflow.math import abs
from ..utils import is_prime

def to_integer(x) :
    """
    This function transforms a real number into an integer and may be used as the
    activation of the last layer of the neural network.

    Parameters
    ----------
    x : float - the last pre-activation

    Returns
    -------
    res : integer - the integer output of the neural network

    """
    
    return floor(abs(x))

def to_binary(x) :
    """
    This function transforms a real number into a binary value 0 or 1 depending
    on the primality of its to_integer transformation. It may be used as the
    activation of the last layer of the neural network.

    Parameters
    ----------
    x : float - the last pre-activation

    Returns
    -------
    res : integer - the integer output of the neural network

    """
    
    return is_prime(to_integer(x))

def initialize_model() :
    """
    This function initializes the deep neural network used for the prime number
    generation.

    Returns
    -------
    model : keras object - the deep neural network model

    """
    
    # Defining our neural network model
    # No bias to avoid approximating a constant function
    model = Sequential([
            InputLayer(input_shape = (1,), dtype = 'int64'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(100, activation = 'relu', use_bias = False, kernel_initializer = 'he_normal'),
            Dense(1, activation = to_integer) # Output layer
        ])

    return model
