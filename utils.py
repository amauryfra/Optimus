"""
This module includes all the utility functions used in the rest of the project.
"""
# Imports
import numpy as np
from random import randint as random_randint
from sympy import isprime

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
        return np.random.randint(low, high + 1, size, dtype = 'int64')
    else :
        # Else -> Probability too high, draw with no replacement 
        # This is longer since all integers are listed beforhand 
        return np.random.choice(np.arange(low, high + 1, dtype = 'int64'), \
                                size, replace = False)
    

def is_prime(n) :
    """
    This function performs a primality test based on the isprime function given 
    in the sympy library.
    
    Parameters
    ----------
    n : integer - the tested integer

    Returns
    -------
    res : integer - 1 if n is prime, 0 otherwise

    """
    
    return int(isprime(n))


def is_prime_array(arr) :
    """
    This function performs a primality test on every element of an array, based 
    on the isprime function given in the sympy library. It works in-place,
    changing the values in the array to 0 or 1 accordingly.
    
    Parameters
    ----------
    arr : K-dimensional numpy array - the tested integers array

    Returns
    -------
    arr : K-dimensional numpy array  - the array is modified in-place, the element
    is set to 1 if prime, 0 otherwise

    """
    
    # Loop over elements in array
    for index, number in enumerate(arr) : 
        if is_prime(number) : # Testing primality
            arr[index] = 1
        else :
            arr[index] = 0       
    return arr


