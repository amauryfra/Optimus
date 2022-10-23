"""
This script includes all the useful functions for the training of the
generative model.
"""

# Imports
import numpy as np
import tensorflow as tf
import keras.backend as K
import copy
import sys
from progress.bar import ChargingBar
sys.path.insert(0, '..')
from utils import is_prime_tensor
from training.model import to_integer



def create_weighted_binary_crossentropy(primes_weights, not_primes_weights) :
    """
    This function returns a weighted binary crossentropy loss function.
    
    Parameters
    ----------
    primes_weights : float - estimate proportion of primes 
    not_primes_weights : float - estimate proportion of not primes 
    
    Returns
    -------
    res : function - the weighted binary crossentropy loss function

    """
    
    def weighted_binary_crossentropy(ground_truth, discriminator_output) :
        """
        This function computes the weighted binary crossentropy loss.
        
        Parameters
        ----------
        ground_truth : K-dimensional tensor  - the ground truth as a 0/1 prime classification
        discriminator_output : K-dimensional tensor - the actual output of the discriminator 
            as a probability of being prime

        Returns
        -------
        res : float - the classification weighted binary crossentropy loss

        """
        
        base_loss = K.binary_crossentropy(ground_truth, discriminator_output)
        weight_vector = ground_truth * primes_weights + (1. - ground_truth) * not_primes_weights
        loss = weight_vector * base_loss
    
        return K.mean(loss)
    
    return weighted_binary_crossentropy



def sampling(ground_truth, latent_ints, wanted_prime_ratio, minimum_batch_size) :
    """
    This function performs undersampling over non prime entries. The number of 
    generated primes over the total number of integers generated must satisfy
    the wanted prime ratio. 
    
    Parameters
    ----------
    ground_truth : K-dimensional tensor  - the ground truth as a 0/1 prime classification
    latent_ints : K-dimensional numpy array - the array of latent integers used as inputs
    wanted_prime_ratio : float - the minimum ratio of primes in the generated batch
    minimum_batch_size : integer - the minimum number of entries in the sampled batch 
    
    Returns
    -------
    res : boolean - True if at least 1 prime has been generated in the batch
    new_latent_ints : N-dimensional numpy array - the undersampled latent integers

    """
    
    nb_latent_ints = latent_ints.shape[0]
    index = np.arange(start = 0, stop = nb_latent_ints)
    
    # Separate primes
    primes_index = tf.where(ground_truth > 0).numpy()
    nb_latent_primes = primes_index.shape[0]
    # Verifying some primes have been generated 
    if nb_latent_primes == 0 :
        return False, None
    
    primes_index = np.reshape(primes_index, (nb_latent_primes,))
    
    # And not primes
    not_primes_index = np.delete(index, primes_index)
    
    # If already all good
    if nb_latent_primes / nb_latent_ints >= wanted_prime_ratio :
        return True, copy.deepcopy(latent_ints)
    
    # How many deletes to obtain the wanted prime ratio in the training batch 
    ideal_nb_del = int(nb_latent_ints - nb_latent_primes / wanted_prime_ratio)
    
    # Is it possible to delete the ideal amount of not primes without obtaining 
    # a batch that is too small 
    if nb_latent_ints - ideal_nb_del > minimum_batch_size :
        nb_del = ideal_nb_del
    else :
        nb_del = nb_latent_ints - minimum_batch_size
    
    # Perform delete
    del_not_primes_idx = np.random.choice(not_primes_index, size = nb_del, replace = False)
    
    # Obtain new latent ints 
    new_latent_ints = copy.deepcopy(latent_ints)
    new_latent_ints = np.delete(new_latent_ints, del_not_primes_idx)

    
    return True, new_latent_ints



def discriminator_loss(disc_base_loss, ground_truth, discriminator_output) :
    """
    This function computes the discriminator model loss.
    
    Parameters
    ----------
    disc_base_loss : function or keras object - the base loss function to compute the loss 
    ground_truth : K-dimensional tensor  - the ground truth as a 0/1 prime classification
    discriminator_output : K-dimensional tensor - the actual output of the discriminator 
        as a probability of being prime
    

    Returns
    -------
    res : float - the classification base loss to be used in Elastic Weight Consolidation 

    """
    
    return disc_base_loss(ground_truth, discriminator_output) 



def generator_loss(gen_base_loss, discriminator_output) :
    """
    This function computes the generator model loss.
    
    Parameters
    ----------
    gen_base_loss : function or keras object - the base loss function to compute the loss 
    discriminator_output : K-dimensional tensor - the actual output of the discriminator 
        evaluated on the generated data, as a probability of being prime
    

    Returns
    -------
    res : float - the classification base loss to be used in Elastic Weight Consolidation 

    """
    # Penalizing small means -> 
    # prime numbers are more concentrated within the small values
    inverse_mean = tf.math.reciprocal_no_nan(tf.math.reduce_mean(discriminator_output))
    inverse_mean = tf.math.square(inverse_mean)
    
    # All outputs are targeted to be primes
    return gen_base_loss(tf.ones_like(discriminator_output), discriminator_output) + inverse_mean
    
  

def compute_fisher(generator, discriminator, for_generator, latent_ints, nb_samples) :
    """
    This function computes the Fisher matrix related to the model on some 
    provided data samples.

    Parameters
    ----------
    generator : keras object - the generator deep neural network model
    discriminator : keras object - the discriminator deep neural network model
    for_generator : boolean - set to True if the generator is used
    latent_ints : N-dimensional numpy array - the array of latent integers used as inputs
    nb_samples : integer - the number of samples to use
    

    Returns
    -------
    fisher_mat : K-dimensional numpy array - the Fisher information matrix elements

    """
    
    # Fisher information matrix for generator
    if for_generator :
        
        # Retreiving weights
        weights = generator.weights
        # Initializing Fisher information matrix/container
        fisher_mat = np.array([np.zeros(layer.numpy().shape) for layer in weights],
                                  dtype = object)
        
        # Building log-likelihood for Fisher information
        for i in range(nb_samples) :
            index = np.random.randint(latent_ints.shape[0])
            with tf.GradientTape() as tape :
                loglikelihood = tf.math.log(discriminator(generator(np.array([latent_ints[index]]))))
        

            # Gradients
            grads = tape.gradient(loglikelihood, weights)
            for m in range(len(weights)) :
                fisher_mat[m] += np.square(grads[m])
        
        
    # Fisher information matrix for discriminator 
    else :
        
        # Retreiving weights
        weights = discriminator.weights
        # Initializing Fisher information matrix/container
        fisher_mat = np.array([np.zeros(layer.numpy().shape) for layer in weights],
                                  dtype = object)
        
        # Building log-likelihood for Fisher information
        for i in range(nb_samples) :
            index = np.random.randint(latent_ints.shape[0])
            with tf.GradientTape() as tape :
                loglikelihood = tf.math.log(discriminator(generator(np.array([latent_ints[index]]))))
                
            # Gradients
            grads = tape.gradient(loglikelihood, weights)
            for m in range(len(weights)) :
                fisher_mat[m] += np.square(grads[m])
                
    fisher_mat /= nb_samples
    return fisher_mat



def get_penalty(model, previous_model, fisher_matrix, lambda_) :
    """
    This function computes the penalty term added to the loss in the Elastic
    Weight Consolidation framework.
    
    Parameters
    ----------
    model : keras object - the deep neural network model
    previous_model : keras object - the previously trained deep neural network model
    fisher_matrix : K-dimensional numpy array - the Fisher information matrix elements
    lambda_ : float - the penalty coefficient 
    

    Returns
    -------
    penalty : float - the penalty term in the loss function 

    """
    
    penalty = 0
    for u, v, w in zip(fisher_matrix, model.weights, previous_model.weights) :
        penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
    return 0.5 * lambda_ * penalty



def train_batch(generator, discriminator,\
                previous_generator, previous_discriminator,\
                gen_optimizer, disc_optimizer, \
                gen_base_loss, disc_base_loss,
                fisher_generator, fisher_discriminator, \
                latent_ints, steps, lambda_, \
                wanted_prime_ratio, minimum_batch_size) :
    """
    This function computes the loss and applies gradient descent to the model.

    Parameters
    ----------
    generator : keras object - the generator deep neural network model
    discriminator : keras object - the discriminator deep neural network model
    previous_generator : keras object - the previously trained generator deep neural network model
    previous_discriminator : keras object - the previously trained discriminator deep neural network model
    gen_optimizer : keras object - the optimizer used to train the generator model
    disc_optimizer : keras object - the optimizer used to train the discriminator model
    gen_base_loss : function - the base loss function to be used for training the generator
    disc_base_loss : function - the base loss function to be used for training the discriminator
    fisher_generator : K-dimensional numpy array - the Fisher information matrix elements of the generator
    fisher_discriminator : K-dimensional numpy array - the Fisher information matrix elements of the discriminator
    latent_ints : N-dimensional numpy array - the array of latent integers used as inputs
    steps : integer - the number of steps used for the gradient descent 
    lambda_ : float - the penalty coefficient 
    wanted_prime_ratio : float - the minimum ratio of primes in the generated batch
    minimum_batch_size : integer - the minimum number of entries in the sampled batch 
    
    
    Returns
    -------
    prime_ratio : float - the ratio of true primes generated over all outputs 
    disc_confusion_matrix : 2x2-dimensional numpy array - the confusion matrix
        related to the discriminator classification 
    res : tensor - the generator outputs after all training steps
    
    """
    
    # Undersampling
    ground_truth = is_prime_tensor(to_integer(generator(latent_ints, training = False)))
    some_primes, sampled_latent_ints = sampling(ground_truth, latent_ints, \
                                                wanted_prime_ratio, minimum_batch_size) 
    # Learning only if at least one prime number has been encountered
    if not some_primes :
        return
    
    # Progress bar
    bar = ChargingBar('Training on current batch', max = steps,\
                      suffix = '%(percent).1f%% - Remaining %(eta)ds')
        
    # Descending several times
    for step in range(steps) :
        
        # Computing ground truth
        ground_truth = is_prime_tensor(to_integer(generator(sampled_latent_ints, training = False)))
        # Adapting data 
        ground_truth = tf.cast(ground_truth, dtype = tf.float32)
        ground_truth = tf.reshape(ground_truth, shape = (ground_truth.shape[0],1))
        
        # Gradient steps
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :
            generator_outputs = generator(sampled_latent_ints, training = True)
            
            discriminator_outputs = discriminator(generator_outputs, training = True)
            
            gen_loss = generator_loss(gen_base_loss, discriminator_outputs) 
            disc_loss = discriminator_loss(disc_base_loss, ground_truth, discriminator_outputs) 
            
            # Elastic Weight Consolidation 
            if fisher_generator is not None and fisher_discriminator is not None :
                gen_loss += get_penalty(generator, previous_generator, \
                                        fisher_generator, lambda_) 
                disc_loss += get_penalty(discriminator, previous_discriminator,\
                                         fisher_discriminator, lambda_) 
                    
        # Applying gradients
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        
        bar.next()
        
    bar.finish()
    
    

def compute_metrics(generator, discriminator, latent_ints) :
    """
    This function computes metrics on the two models. 

    Parameters
    ----------
    generator : keras object - the generator deep neural network model
    discriminator : keras object - the discriminator deep neural network model
    latent_ints : N-dimensional numpy array - the array of latent integers used as inputs
    
    
    Returns
    -------
    prime_ratio : float - the ratio of true primes generated over all outputs 
    prime_ratio_unique : float - the ratio of true unique primes generated over all outputs 
    disc_confusion_matrix : 2x2-dimensional numpy array - the confusion matrix
        related to the discriminator classification 
    maybe_primes_int : N-dimensional numpy array  - the generator outputs
    maybe_primes_unique : K-dimensional numpy array  - the unique generator outputs
    
    """
    
    # Computing metrics
    maybe_primes = generator(latent_ints, training = False)
    maybe_primes_int = to_integer(maybe_primes)
    are_primes = is_prime_tensor(maybe_primes_int)
    maybe_primes_unique = np.unique(maybe_primes_int)
    are_primes_unique = is_prime_tensor(maybe_primes_unique)
    
    # Ratio of correctly generated primes (non unique)
    prime_ratio = tf.math.count_nonzero(tf.greater_equal(are_primes, 1))
    prime_ratio /= tf.size(are_primes, out_type = tf.int64)
    
    # Ratio of correctly generated primes (unique)
    prime_ratio_unique = tf.math.count_nonzero(tf.greater_equal(are_primes_unique, 1))
    prime_ratio_unique /= tf.size(are_primes_unique, out_type = tf.int64)
    
    # Confusion matrix related to discriminator classification
    are_primes = tf.cast(are_primes, dtype = tf.float32)
    discriminator_outputs = discriminator(maybe_primes, training = False)
    discriminator_outputs_confusion = tf.reshape(discriminator_outputs, \
                                                 shape = (discriminator_outputs.shape[0],))
    discriminator_outputs_confusion = tf.round(discriminator_outputs_confusion)
    # Confusion matrix related to discriminator classification 
    # Possible overflow -> To fix!
    try : 
        disc_confusion_matrix = tf.math.confusion_matrix(are_primes, discriminator_outputs_confusion, \
                                                         num_classes = 2, dtype = tf.uint64)
    except :
        disc_confusion_matrix = None
        
    return prime_ratio.numpy(), prime_ratio_unique.numpy(), \
        disc_confusion_matrix.numpy(), maybe_primes_int.numpy(), maybe_primes_unique
    
            
