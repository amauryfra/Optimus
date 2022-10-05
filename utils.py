"""
This module includes all the utility functions used in the rest of the project.
"""
# Imports
from random import randint
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
    
    return randint(low, high)


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
