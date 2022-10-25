"""
This script includes all the utility functions used in the rest of the project.
"""

# Imports
import numpy as np
from random import randint as random_randint
from sympy import isprime
import tensorflow as tf



def random_integer(low, high) :
    """
    This function generates a random integer low <= n <= high, using a uniform
    distribution, via the random library.

    Parameters
    ----------
    low : integer - the lower bound value
    high : integer - the upper bound value

    Returns
    -------
    n : integer - a random integer

    """

    return random_randint(low, high)



def random_vector_integers(low, high, size) :
    """
    This function generates a random vector of unique integers between low and
    high, using a uniform distribution, via the numpy.random library.

    Parameters
    ----------
    low : integer - the lower bound value
    high : integer - the upper bound value
    size : integer - the size of the random vector

    Returns
    -------
    vec : size-dimensional numpy array - a random vector of unique integers

    """

    # Monitoring probability of having two integers equal in the list
    if 1 - np.exp(-((size-1) * size) / (2 * (high-low))) < 1e-3 :
        # If probability small enough -> draw from uniform distribution 
        return np.random.randint(low, high + 1, size, dtype = 'uint64')
    else :
        # Else -> Probability too high, draw with no replacement
        # This is longer since all integers are listed beforhand
        return np.random.choice(np.arange(low, high + 1, dtype = 'uint64'), \
                                size, replace = False)



def var_access(n) :
    """
    This function is a workaround to access the data inside the 1-dimensional
    tensor passed as input. It performs the same process as tensor.numpy() in
    tensorflow 2.0. As said method is not working in our tf.map_fn implementation,
    we use the string representation of the tensor to access the stored integer
    instead.

    Parameters
    ----------
    n : 1-dimensional tensor - the tensor containing the target integer

    Returns
    -------
    res : integer - the value of the accessed integer as a python int

    """

    # We use the string representation of the tensor object
    string = repr(tf.squeeze(n))
    # Example of output :  '<tf.Tensor: shape=(), dtype=uint64, numpy=14>'
    # We aim at returning 14 as an int

    # The first string printed is not relevant
    # We manage this case separately
    if string[-7:-1] == 'uint64' :
        return None

    # The target int starts after 'numpy=' and ends before '>'
    string = string[42 : len(string) - 1]
    # The Answer to the Ultimate Question of Life, the Universe, and Everything
    return int(string)



def is_prime(n) :
    """
    This function performs a primality test based on the isprime function given
    in the sympy library.

    Parameters
    ----------
    n : 1-dimensional tensor - the tested integer

    Returns
    -------
    res : 1-dimensional tensor - value is 1 if n is prime, 0 otherwise

    """

    # The first element returned using the workaround is not relevant
    # We manage this case separately
    if var_access(n) == None :
        return tf.constant(-1000, dtype = tf.int8)
    # This exception does not interfere with the training process
    # -1000 does not appear in the predictions tensor

    # Primality test in tensor
    return tf.cast(tf.constant(isprime(var_access(n))), dtype = tf.int8)



def is_prime_tensor(arr) :
    """
    This function performs a primality test on every element of a tensor, based
    on the isprime function given in the sympy library.

    Parameters
    ----------
    arr : K-dimensional tensor - the tested integers tensor

    Returns
    -------
    arr : K-dimensional tensor  - the element is set to 1 if prime,
        0 otherwise

    """

    return tf.map_fn(is_prime, arr, tf.int8)



def confusion_dict(confusion_matrix) :
    """
    This function transforms the confusion matrix given as a numpy array into
    a dictionary.

    Parameters
    ----------
    confusion_matrix : 2x2-dimensional numpy array - the confusion matrix

    Returns
    -------
    c_dict : dictionary - the confusion matrix stated as a dictionnary

    """
    c_dict = {}
    c_dict['Primes classified as primes'] = confusion_matrix[1,1]
    c_dict['Primes classified as not primes'] = confusion_matrix[1,0]
    c_dict['Not primes classified as not primes'] = confusion_matrix[0,0]
    c_dict['Not primes classified as primes'] = confusion_matrix[0,1]

    return c_dict
