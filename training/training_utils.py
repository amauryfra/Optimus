"""
This module includes all the useful functions for the training of the
generative model.
"""

# Imports
import numpy as np
import tensorflow as tf

def compute_fisher(model, data_samples, nb_samples) :
    """
    This function computes the Fisher matrix related to the model and some 
    provided data samples.

    Parameters
    ----------
    model : keras object - the deep neural network model
    data_samples : N-dimensional numpy array - the data samples used to compute 
    the matrix
    nb_samples : integer - the number of samples to use
    

    Returns
    -------
    fisher_mat : K-dimensional numpy array - the Fisher information matrix elements

    """
    
    # Retreiving weights
    weights = model.weights
    # Initializing Fisher information matrix/container
    fisher_mat = np.array([np.zeros(layer.numpy().shape) for layer in weights],
                           dtype = object)

    # Building Fisher information
    for i in range(nb_samples) :
        index = np.random.randint(data_samples.shape[0])
        with tf.GradientTape() as tape :
             logits = tf.nn.log_softmax(model(np.array([data_samples[index]])))
        grads = tape.gradient(logits, weights)
        for m in range(len(weights)) :
            fisher_mat[m] += np.square(grads[m])

    fisher_mat /= nb_samples
    return fisher_mat

def get_penalty(model, previous_model, fisher_matrix, lbda) :
    """
    This function computes the Fisher matrix related to the model and some 
    provided data samples.

    Parameters
    ----------
    model : keras object - the deep neural network model
    previous_model : keras object - the previously trained deep neural network model
    fisher_matrix : K-dimensional numpy array - the Fisher information matrix elements
    lbda : float - the penalty coefficient 
    

    Returns
    -------
    penalty : float - the penalty contribution in the loss function 

    """
    
    penalty = 0
    for u, v, w in zip(fisher_matrix, model.weights, previous_model.weights) :
        penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
    return 0.5 * lbda * penalty

