from tf_helpers import *
from saferl import *

class ConstrainedProximalPolicyOptimization(ConstrainedAgent):
    """ An RL agent for CMDPs which does random action choices """
    def __init__(self, env, sess, epsilon = 0.1):
        """ Initialize for environment
             eps: policy clipping
             delta: when to stop iterations, max KL-divergence
        """
        super(ConstrainedRandomAgent, self).__init__(env)
        self.epsilon = epsilon
        self.delta = delta

        def g(eps, A):
            """ function g(eps, A), see PPO def. above """
            def step_fcn(x):
                return (tf.math.sign(x) + 1) / 2.
            return tf.multiply(step_fcn(A), A * (1 + epsilon)) + tf.multiply(step_fcn(-A), A * (1 - epsilon))

        # states
        self.p_states = tf.placeholder(tf.float64, shape = (None, S_DIM,))

        # taken actions
        self.p_actions = tf.placeholder(tf.int64, shape = (None,))

        # rewards obtained
        self.p_rewards = tf.placeholder(tf.float64, shape = (None,))

        # maximal constraint return
        self.p_threshold = tf.placeholder(tf.float64)

        # costs obtained
        self.p_disc_costs = tf.placeholder(tf.float64, shape = (None,))

        # discounted rewards obtained
        self.p_discounted_rewards_to_go = tf.placeholder(tf.float64, shape = (None,))

        # state is an input to the network
        z = self.p_states

        # some fully connected stuff
        z = fc_layer(z, 10)

        # some fully connected stuff
        #z = fc_layer(z, 10)

        # POLICY network head
        with tf.name_scope('policy_layers'):
            #z = fc_layer(z, 10)
            z_policy = fc_layer(z, 10)
            z_policy = fc_layer(z_policy, ACTIONS, activation = None)
            self.t_logits_policy = tf.nn.softmax(z_policy)
            # predicted labels
            self.t_labels = tf.argmax(logits_policy, axis = 1)
    
        # VALUE network head
        with tf.name_scope('value_layers'):
            z_value = fc_layer(z, 10)
            self.t_value = fc_layer(z_value, 1, activation = None)

# next value
value_next = tf.concat([value[1:, :], [[0]]], axis = 0)

# advantage function
advantage = rewards + tf.reshape(gamma_discount * value_next - value, (-1,))

# Loss 2
L2 = tf.reduce_mean(tf.square(tf.reshape(value, (-1,)) - discounted_rewards_to_go))

# one-hot encoded actions
a_one_hot = tf.one_hot(actions, ACTIONS)

# taken logits
#logits_taken = tf.gather(logits, actions, axis = 1)
logits_taken = tf.boolean_mask(logits_policy, a_one_hot)

# pi_theta / pi_theta_k
pi_theta_pi_thetak = tf.divide(logits_taken, tf.stop_gradient(logits_taken))
advantage_nograd = tf.stop_gradient(advantage)
part1 = tf.multiply(pi_theta_pi_thetak, advantage_nograd)
part2 =                      g(epsilon, advantage_nograd)

# calculated loss
L1 = -tf.reduce_mean(tf.minimum(part1, part2))

# discounted constraint return
J_C = disc_costs[0]

# all parameters
params = tf.trainable_variables()

# taken logits log
log_logits = tf.log(logits_taken)

# constraint function for reward min
constraint_return_int = tf.reduce_sum(tf.multiply(log_logits, disc_costs))

# gradient of the CONSTRAINT
g_C = tf.gradients(constraint_return_int, params)

# gradient of the REWARD (PPO function)
g_R = tf.gradients(-L1, params)

# goal to constraint gradient cosine
RtoC = cos_similarity(g_C, g_R)

# TOTAL LOSS
loss = L1 + L2

# variable BEFORE the step
theta_0 = [tf.Variable(tf.zeros_like(p)) for p in params]

# current parameters
theta_1 = params

# save theta1 -> theta0
save_to0 = tf.group([a.assign(b) for a, b in zip(theta_0, theta_1)])

# OPTIMIZING after saving theta to theta_0
with tf.control_dependencies([save_to0]):
    opt1 = tf.train.AdamOptimizer(0.001).minimize(L1)
    opt2 = tf.train.AdamOptimizer(0.001).minimize(L2)
    # one learning iteration
    opt_step = tf.group([opt1, opt2])

# non-zero denominator
eps_ = 1e-5

# assigning theta projection after optimizing
with tf.control_dependencies([opt_step]):
    
    # SLACK to go
    R = C_max - J_C + tf.reduce_sum(
        [tf.reduce_sum(tf.multiply(t0 - t1, g)) for t0, t1, g in zip(theta_0, theta_1, g_C) if g is not None])

    # negative number if violated, zero if OK
    R_clipped = -tf.nn.relu(-R)
    
    project_step = tf.group([t.assign(t + g * R_clipped / (eps_ + norm_fro_sq(g))) for t, g in zip(params, g_C) if g is not None])

step = tf.group([save_to0, opt_step, project_step])
#step = opt_step


    def sample_action(self, observation):
        """ Sample an action given observation, typically runs on a GPU """
        p = self.sess.run(self.p_logits_policy, feed_dict = {self.t_states: [observation]})[0]
        return np.random.choice(range(self.n_actions), p = p)

    def episode_start(self):
        """ Called each time a new episode starts """
        pass

    def episode_end(self):
        """ Called each time an episode ends """
        pass

    def process_feedback(self, state, reward, cost, state_new, done, info):
        """ Called inside the train loop, typically just stores the data """
        pass

    def train_start(self):
        """ Called before one training phase """
        pass

    def train(self):
        """ Train method, typically runs on a GPU """
        pass

