"""
This script performs the training of the prime generator model.
"""

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
assert tf.executing_eagerly() == True
tf.get_logger().setLevel('ERROR')
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import initialize_generator
from model import initialize_discriminator
from training_utils import train_batch
from training_utils import compute_fisher
from tensorflow.math import abs as abs_tf
import keras
keras.utils.get_custom_objects().update({'abs': abs_tf})
from keras.models import clone_model
import os.path
import time
import sys
sys.path.insert(0, '..')
from utils import random_vector_integers
from test import test_models




##### Setting training hyperparameters #####
nb_batches = 2 #450 
batch_size = 2000

low_latent = 1
high_latent = 1000000

wanted_prime_ratio = 0.35
minimum_batch_size = 100

disc_base_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing = True, 
    alpha = 0.95, gamma = 2.0, from_logits = False)
gen_base_loss =  tf.keras.losses.BinaryCrossentropy(from_logits = False)

steps = 25
lambda_ = 1.0
lr = 1e-4
nb_samples = 75

test_session_occurrences = 10

##### .............................. #####



def train() :
    """
    This function implements the main continual training loop of the adversarial 
    training scheme.
    
    Returns
    -------
    prime_ratio_array : N-dimensional numpy array  - array containing the generated 
        primes ratio computed at each test sessions
    prime_ratio_unique_array : N-dimensional numpy array  - array containing the 
        unique generated primes ratio computed at each test sessions
    avg_prime_probability_array : N-dimensional numpy array  - array containing the
        probability of randomly drawing primes in between the minimum and maximum 
        outputs computed at each test sessions
    confusion_list : list of dictionaries - list containing the confusion dictionaries
        computed at each test sessions
    
    """
    
    # Initializing models
    generator = initialize_generator()
    discriminator = initialize_discriminator()
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    # Initializing metrics container
    test_session = -1
    prime_ratio_array = np.zeros(nb_batches // test_session_occurrences + 1)
    prime_ratio_unique_array = np.zeros(nb_batches // test_session_occurrences + 1)
    avg_prime_probability_array = np.zeros(nb_batches // test_session_occurrences + 1)
    confusion_list = []
    
    # Loading previous models
    is_previous = False
    if os.path.isfile('Optimus_Generator.h5') :
        is_previous = True
        generator.load_weights('Optimus_Generator.h5')
        discriminator.load_weights('Optimus_Discriminator.h5')
        fisher_generator = np.load('fisher_generator.npy', allow_pickle = True)
        fisher_discriminator = np.load('fisher_discriminator.npy', allow_pickle = True)
        previous_generator = clone_model(generator)
        previous_generator.set_weights(generator.get_weights())
        previous_discriminator = clone_model(discriminator)
        previous_discriminator.set_weights(discriminator.get_weights())
        

    # Training loop
    latent_ints = None
    with tf.device('/gpu:0') :
        for batch_index in range(nb_batches) :
            
            # Prints
            print('')
            print('Starting training of batch : ', batch_index + 1)
            
            if not is_previous and batch_index == 0 :
                # For first start 
                previous_generator = None
                previous_discriminator = None
                fisher_generator = None
                fisher_discriminator = None
            
            if batch_index > 0 : 
                # Cloning
                previous_generator = clone_model(generator)
                previous_generator.set_weights(generator.get_weights())
                previous_discriminator = clone_model(discriminator)
                previous_discriminator.set_weights(discriminator.get_weights())
                # Getting Fisher information matrix
                fisher_generator = compute_fisher(generator, discriminator, \
                                                  True, latent_ints, nb_samples) 
                fisher_discriminator = compute_fisher(generator, discriminator, \
                                                      False, latent_ints, nb_samples)
            
            # Draw latent integers
            latent_ints = random_vector_integers(low = low_latent, high = high_latent, \
                                                     size = batch_size)
            
            # Perform training on latent integers
            t0 = time.time()
            train_batch(generator, discriminator,\
                            previous_generator, previous_discriminator,\
                            gen_optimizer, disc_optimizer, \
                            gen_base_loss, disc_base_loss,
                            fisher_generator, fisher_discriminator, \
                            latent_ints, steps, lambda_, \
                            wanted_prime_ratio, minimum_batch_size)
                
            # Save model weights
            generator.save('Optimus_Generator.h5')
            discriminator.save('Optimus_Discriminator.h5')
            if batch_index > 0 :
                np.save('fisher_generator.npy', fisher_generator)
                np.save('fisher_discriminator.npy', fisher_discriminator)
            
            # Prints
            t1 = time.time()
            print('')
            print('End of training on current batch | Performed in : ', \
                  format(t1-t0, '.2f'), ' seconds')
            print('')
                
            # Test sessions
            if batch_index % test_session_occurrences == 0 :
                test_session += 1
                print('')
                print('------------------------->  Test session : ', test_session + 1,\
                      ' <-------------------------')
                
                
                prime_ratio, prime_ratio_unique, \
                avg_prime_probability, confusion, \
                _, _ = test_models(generator, discriminator, verbose = 3)
                
                prime_ratio_array[test_session] = prime_ratio
                prime_ratio_unique_array[test_session] = prime_ratio_unique
                avg_prime_probability_array[test_session] = avg_prime_probability
                confusion_list += [confusion]
    
    
    # Prime ratio evolution plot
    plt.figure(figsize = (20, 10), dpi = 60)
    plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
    x_axis = test_session_occurrences * np.arange(0, prime_ratio_array.shape[0]) + 1
    plt.plot(x_axis, prime_ratio_unique_array, label = 'Generated primes ratio (unique)')
    plt.plot(x_axis, avg_prime_probability_array, label = 'Probability of drawing a prime in a similar range')
    plt.legend(loc = 'best')
    plt.xlabel('Number of batches fed for training')
    plt.ylabel('Ratios')
    plt.title('Evolution of performance of the generator model')
    
    print('................................End of training................................')
    print('...............................................................................')
    print('')
    
    return prime_ratio_array, prime_ratio_unique_array, avg_prime_probability_array, confusion_list


if __name__ == "__main__" :
    train()
    
    
    