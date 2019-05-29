import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

def fc_layer(x, n, activation = tf.nn.relu):
    """ Fully connected layer for input x and output dim n """
    return tf.contrib.layers.fully_connected(x, n, activation_fn=activation,
    weights_initializer=tf.initializers.lecun_normal(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, trainable=True)

class ExperienceReplay():
    def __init__(self, N = 1000):
        """ Create experience buffer with capacity N """
        self.N = N
        self.buf = []
    def store(self, exp):
        """ Store one experience """
        self.buf.append(exp)
        
        # removing extra elements
        extra = len(self.buf) - self.N
        if extra > 0:
            self.buf = self.buf[extra:]
            
    def sample(self, how_many = 10):
        """ Sample a mini-batch from the buffer """
        assert len(self.buf) > 0, "Experience buffer is empty, cannot sample"
        indices = np.random.choice(len(self.buf), how_many)
        result = [self.buf[i] for i in indices]
        return result

def dot_and_sum1(Z1, Z2):
    """ Return a vector of length Z1.shape[0] == Z2.shape[0] with i'th entry being sum of Z1ik*Z2ik """
    return tf.reduce_sum(tf.multiply(Z1, Z2), axis = 1)

def mse(x, y):
    """ Mean squared error tensor """
    return tf.reduce_mean(tf.square(tf.cast(x, tf.float64) - tf.cast(y, tf.float64)))

def soft_bellman_loss(a_one_hot, rewards, q_values_from_to, logits_policy, need_next_q, gamma_discount = 0.0):
    """ Bellman residual w.r.t. actions, rewards and q-values """
    # q-values from-to, equal length
    q_values_from = q_values_from_to[0::2, :]
    q_values_to = q_values_from_to[1::2, :]

    # taken Q value (from)
    q_taken = tf.boolean_mask(q_values_from, a_one_hot)

    # maximal Q value
    # TODO: replace with soft (use logits!)
    #q_max   = tf.reduce_max(tf.stop_gradient(q_values_to), axis = 1)
    q_soft = dot_and_sum1(q_values_to, logits_policy)
    
    # loss tensor (only optimizing over q_taken!)
    loss = mse(rewards + gamma_discount * tf.multiply(need_next_q, tf.stop_gradient(q_soft)), q_taken)
    return loss

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

def replay_test():
    replay = ExperienceReplay()
    replay.store([1,2])
    replay.store([3,4])
    rs = replay.sample()
    assert [1,2] in rs or [3,4] in rs
    
    replay = ExperienceReplay(N = 3)

    replay.store(1)
    replay.store(2)
    replay.store(3)
    replay.store(4)
    replay.store(5)
    replay.store(6)

    assert replay.buf == [4, 5, 6]
replay_test()

# Jensen-Shannon Divergence
# D_{JS}(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
# M = 0.5 * (P + Q) -> normalized!
def kl(x, y):
    """ KL-divergence """
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)
def jsd(x, y):
    M = 0.5 * (x + y)
    return 0.5 * kl(x, M) + 0.5 * kl(y, M)

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

def get_current_eps(eps_decay, iteration):
    # finding minimal key s.t. <= iteration
    for key in sorted(eps_decay.keys(), reverse = True):
        if key <= iteration:
            return eps_decay[key]
    print("Error: could not find eps")
    return None

def test_decay():
    eps_decay = {0: 0.9, 100: 0.5, 200: 0.3, 300: 0.1, 500: 0}
    assert get_current_eps(eps_decay, 499) == 0.1
    assert get_current_eps(eps_decay, 1000) == 0
    assert get_current_eps(eps_decay, 500) == 0
    assert get_current_eps(eps_decay, 100) == 0.5
    assert get_current_eps(eps_decay, 0) == 0.9
test_decay()

def plot_performance(r, c):
    plt.figure()
    plt.title('DQN (safe) on CartPole')
    plt.plot(pd.DataFrame(r).rolling(20).median(), label = 'reward')
    plt.plot(pd.DataFrame(c).rolling(20).median(), label = 'cost')
    plt.xlabel('Episode')
    plt.ylabel('Reward/Cost')
    #plt.savefig('sdqn.eps', bbox_inches = 'tight')
    plt.legend()
    plt.show()

def plot_decay(eps_decay):
    plt.title('Epsilon greedy decay')
    plt.ylabel('eps for eps-greedy')
    plt.xlabel('Iteration')
    plt.plot(eps_decay.keys(), eps_decay.values())
    plt.show()
