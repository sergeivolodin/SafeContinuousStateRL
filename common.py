import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

def fc_layer(x, n, activation = tf.nn.relu):
    """ Fully connected layer for input x and output dim n """
    return tf.contrib.layers.fully_connected(x, n, activation_fn=activation,
    weights_initializer=tf.initializers.lecun_normal(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, trainable=True)

def mse(x, y):
    """ Mean squared error tensor """
    return tf.reduce_mean(tf.square(tf.cast(x, tf.float64) - tf.cast(y, tf.float64)))

def cost(obs):
    """ Calculate scalar cost of one observation """
    assert isinstance(obs, np.ndarray) and obs.shape == (4,), "Input must be an np-array [x xdot phi phidot]"
    
    # parsing input
    x, x_dot, phi, phi_dot = obs
    
    #X_MAX = 1.0
    #X_DOT_MAX = 0.5
    #PHI_MAX = 0.1
    #PHI_DOT_MAX = 0.5
    
    #if x < 0 or phi < 0:
    #    return 1
    if x > 0: return 1
    
    #if np.any(np.abs([x, x_dot, phi, phi_dot]) > [X_MAX, X_DOT_MAX, PHI_MAX, PHI_DOT_MAX]):
    #    return 1
    
    # in all other cases no cost
    return 0

def loss_to_opt(lst, name = 'some_optimizer'):
    """ List of losses -> group optimize operation """
    return tf.group([tf.train.AdamOptimizer(0.1).minimize(loss) for loss in lst])

def interleave(A, B):
    """ Interleave two lists A, B: a1b1a2b2... """
    # https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python
    assert len(A) == len(B), "Must have arrays of same length"
    return [val for tup in zip(A, B) for val in tup]
assert interleave([1,2,3],[4,5,6]) == [1, 4, 2, 5, 3, 6]

def q_like_function(z, name = '__unknown__', ACTIONS = None, temperature = 1.0):
    # Q network head
    with tf.name_scope('q_layers_' + name):
        
        # state is an input to the network
        #z = states

        # some fully connected stuff
        z = fc_layer(z, 20)

        # some fully connected stuff
        #z = fc_layer(z, 10)
        
        #z = fc_layer(z, 10)
        #z_policy = fc_layer(z, 10)
        z_policy = z
        z_policy = fc_layer(z_policy, ACTIONS, activation = None)
        q_values = z_policy
        #logits_policy = tf.nn.softmax(z_policy)
        # predicted labels
        logits_policy = tf.nn.softmax(temperature * q_values)
    return q_values, logits_policy

def plot_performance(r, c, name = 'DQN (safe) on CartPole', fig = 'sdqn'):
    plt.figure()
    plt.title(name)
    plt.plot(pd.DataFrame(r).rolling(20).median(), label = 'reward')
    plt.plot(pd.DataFrame(c).rolling(20).median(), label = 'cost')
    plt.xlabel('Episode')
    plt.ylabel('Reward/Cost')
    #plt.savefig(fig, bbox_inches = 'tight')
    plt.legend()
    plt.show()
