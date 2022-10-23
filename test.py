"""
This script provides a function to test the prime generator.
"""

# Imports
import numpy as np
from sympy.ntheory.generate import primepi
from utils import confusion_dict
from utils import random_vector_integers
from utils import is_prime_tensor
from training.training_utils import compute_metrics
from training import model



def test_models(generator = None, discriminator = None, verbose = 0) :
    """
    This function tests the trained generator and discriminator.
    
    Parameters
    ----------
    generator : keras object - the generator deep neural network model
    discriminator : keras object - the discriminator deep neural network model
    verbose : integer - controls what information is printed -> 0 for no prints 
        -> 3 for all available test information to be printed
    
    Returns
    -------
    prime_ratio : float - the ratio of true primes generated over all outputs 
    prime_ratio_unique : float - the ratio of unique true primes generated over all outputs 
    avg_prime_probability : float - the probability of randomly drawing a prime number 
        in between the minimum and maximum outputs
    confusion : dictionary - the confusion matrix as a dictionary 
        related to the discriminator classification 
    generated_ints : N-dimensional numpy array - a batch of generator outputs 
    generated_primes : N-dimensional numpy array - a batch of unique generated primes

    """
    
    # Loading models 
    if generator == None :
        generator = model.initialize_generator()
        discriminator = model.initialize_discriminator()
        generator.load_weights('training/Optimus_Generator.h5')
        discriminator.load_weights('training/Optimus_Discriminator.h5')
    
    # Draw latent integers
    latent_ints = random_vector_integers(low = 1, high = int(1e7), \
                                             size = 1000)

    # Compute metrics
    prime_ratio, \
    prime_ratio_unique, \
    disc_confusion_matrix, \
    generated_ints, \
    generated_ints_unique =  compute_metrics(generator, discriminator, latent_ints)

    # Confusion dictionary 
    if disc_confusion_matrix is not None :
        confusion = confusion_dict(disc_confusion_matrix)
    
    # Generator outputs for given latent integers
    generated_ints = generated_ints.reshape((generated_ints.shape[0],))
    generated_primes = generated_ints_unique[is_prime_tensor(generated_ints_unique) > 0]

    # Average probability if random draw of primes
    min_generated_nb = np.min(generated_ints)
    max_generated_nb = np.max(generated_ints)
    primepi_min = int(primepi(min_generated_nb))
    primepi_max = int(primepi(max_generated_nb))
    avg_prime_probability = 2. * (primepi_max - primepi_min) / (max_generated_nb - min_generated_nb + 1)
    
    # Prints
    if verbose > 0 :
        
        print('')
        print('------------------- Test results -------------------')
        print('')
        
        print('Generator prime ratio (unique) : ', format(prime_ratio_unique * 100, '.2f'), '%')
        print('Probability of drawing a prime in same range : ', format(avg_prime_probability * 100, '.2f'), '%')
        print('')
        
        if verbose > 1 :
            
            print('Discriminator confusions : ')
            print('............................')
            if disc_confusion_matrix is not None :
                [print(key,':',value) for key, value in confusion.items()]
            print('')
            
            if verbose > 2 :
                
                print('Batch of generated numbers : ')
                print('............................')
                print(np.random.choice(generated_ints, size = np.min([generated_ints.shape[0], 25]), replace = False))
                print('')
                print('Batch of generated primes : ')
                print('............................')
                print(np.random.choice(generated_primes, size = np.min([generated_primes.shape[0], 25]), replace = False))
                print('')
    
        print('-----------------------------------------------------')
        print('')

    
    return prime_ratio, prime_ratio_unique, avg_prime_probability, confusion, \
        generated_ints, generated_primes



if __name__ == '__main__' :
    _, _, _, _, _, _ = test_models(verbose = 3)
    
    