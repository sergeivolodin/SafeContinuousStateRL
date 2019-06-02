from tf_helpers import *
from saferl import *

class ConstrainedProximalPolicyOptimization(ConstrainedAgent):
    """ An RL agent for CMDPs which does random action choices """
    def __init__(self, env, sess, epsilon = 0.1, delta = 0.01, ignore_constraint = False, steps = 1, lr_policy = 0.01, lr_value = 0.01):
        """ Initialize for environment
             eps: policy clipping
             delta: when to stop iterations, max KL-divergence (not implemented yet)
             steps: number of steps to take in train()
             lr_policy: parameter for Adam (policy opt)
             lr_value: value network learning rate for SGD
        """
        super(ConstrainedProximalPolicyOptimization, self).__init__(env)
        self.epsilon = epsilon
        self.delta = delta
        self.sess = sess
        self.steps = steps
        self.lr_policy = lr_policy
        self.lr_value = lr_value

        def g(eps, A):
            """ function g(eps, A), see PPO def. above """
            def step_fcn(x):
                return (tf.math.sign(x) + 1) / 2.
            return tf.multiply(step_fcn(A), A * (1 + self.epsilon)) + tf.multiply(step_fcn(-A), A * (1 - self.epsilon))

        # states
        self.p_states = tf.placeholder(tf.float64, shape = (None, self.state_dim,), name = 'states')

        # taken actions
        self.p_actions = tf.placeholder(tf.int64, shape = (None,), name = 'actions')

        # rewards obtained
        self.p_rewards = tf.placeholder(tf.float64, shape = (None,), name = 'rewards')

        # maximal constraint return
        self.p_threshold = tf.placeholder(tf.float64, shape = (), name = 'c_threshold')

        # costs obtained
        self.p_disc_costs = tf.placeholder(tf.float64, shape = (None,), name = 'discounted_costs')

        # discounted rewards obtained
        self.p_discounted_rewards_to_go = tf.placeholder(tf.float64, shape = (None,), name = 'discounted_rewards_to_go')

        # constraint return
        self.p_constraint_return = tf.placeholder(tf.float64, shape = (), name = 'constraint_return')

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
            z_policy = fc_layer(z_policy, self.n_actions, activation = None)
            self.t_logits_policy = tf.nn.softmax(z_policy, name = 'logits_policy')
            # predicted labels
            self.t_labels = tf.argmax(self.t_logits_policy, axis = 1)
    
        # VALUE network head
        with tf.name_scope('value_layers'):
            z_value = fc_layer(z, 10)
            self.t_value = fc_layer(z_value, 1, activation = None, name = 'value')

        # next value
        self.t_value_next = tf.concat([self.t_value[1:, :], [[0]]], axis = 0, name = 'next_value')

        # advantage function
        self.t_advantage = tf.identity(self.p_rewards + tf.reshape(self.gamma * self.t_value_next - self.t_value, (-1,)), name = 'advantage')

        # Loss 2
        self.t_L2 = tf.reduce_mean(tf.square(tf.reshape(self.t_value, (-1,)) - self.p_discounted_rewards_to_go), name = 'L2')

        # one-hot encoded actions
        self.t_a_one_hot = tf.one_hot(self.p_actions, self.n_actions, name = 'a_one_hot')

        # taken logits
        #logits_taken = tf.gather(logits, actions, axis = 1)
        self.t_logits_taken = tf.boolean_mask(self.t_logits_policy, self.t_a_one_hot, name = 'logits_taken')

        # pi_theta / pi_theta_k
        pi_theta_pi_thetak = tf.divide(self.t_logits_taken, tf.stop_gradient(self.t_logits_taken), name = 'pi_pi_ratio')
        advantage_nograd = tf.stop_gradient(self.t_advantage)
        part1 = tf.multiply(pi_theta_pi_thetak, advantage_nograd)
        part2 =                      g(self.epsilon, advantage_nograd)

        # calculated loss
        self.t_L1 = tf.reduce_mean(-tf.minimum(part1, part2), name = 'L1')

        # discounted constraint return
        self.t_J_C = self.p_constraint_return#self.p_disc_costs[0]

        # all parameters
        self.params = trainable_of(self.t_L1 + self.t_L2)

        # taken logits log
        self.t_log_logits = tf.log(self.t_logits_taken, name = 'log_logits')

        # constraint function for reward min
        self.t_constraint_return_int = tf.reduce_sum(tf.multiply(self.t_log_logits, self.p_disc_costs), name = 'constr_ret_loss')

        # gradient of the CONSTRAINT
        self.t_g_C = tf.gradients(self.t_constraint_return_int, self.params, name = 'g_C')

        # gradient of the REWARD (PPO function)
        self.t_g_R = tf.gradients(-self.t_L1, self.params, name = 'g_R')

        # goal to constraint gradient cosine
        self.t_RtoC = cos_similarity(self.t_g_C, self.t_g_R, name = 'RtoC')

        # TOTAL LOSS
        self.t_loss = self.t_L1 + self.t_L2

        # variable BEFORE the step
        self.t_theta_0 = [tf.Variable(tf.zeros_like(p), trainable = False) for p in self.params]

        # current parameters
        self.t_theta_1 = self.params

        # save theta1 -> theta0
        self.op_save_to0 = tf.group([a.assign(b) for a, b in zip(self.t_theta_0, self.t_theta_1)])

        # OPTIMIZING after saving theta to theta_0
        with tf.control_dependencies([self.op_save_to0]):
            self.op_opt1 = tf.train.GradientDescentOptimizer(self.lr_policy).minimize(self.t_L1)
            self.op_opt2 = tf.train.GradientDescentOptimizer(self.lr_value).minimize(self.t_L2)
            # one learning iteration
            self.op_opt_step = tf.group([self.op_opt1, self.op_opt2])

        # buffer for experience
        self.buffer = []

        # doing unsafe if there is this flag
        if ignore_constraint:
            self.op_step = self.op_opt_step
            return

        # non-zero denominator
        eps_ = 1e-3

        # assigning theta projection after optimizing
        with tf.control_dependencies([self.op_opt_step]):
            
            # SLACK to go
            self.t_R = tf.identity(self.threshold - self.t_J_C + tf.reduce_sum(
                [tf.reduce_sum(tf.multiply(t0 - t1, g)) for t0, t1, g in zip(self.t_theta_0, self.t_theta_1, self.t_g_C) if g is not None]), name = 'Slack')

            # negative number if violated, zero if OK
            self.t_R_clipped = tf.identity(-tf.nn.relu(-self.t_R), name = 'slack_clipped')
            
            # projection
            self.op_project_step = tf.group([t.assign(t + g * self.t_R_clipped / (eps_ + norm_fro_sq(g))) for t, g in zip(self.params, self.t_g_C) if g is not None])

        self.op_step = tf.group([self.op_save_to0, self.op_opt_step, self.op_project_step])
        #step = opt_step


    def sample_action(self, observation):
        """ Sample an action given observation, typically runs on a GPU """
        p = self.sess.run(self.t_logits_policy, feed_dict = {self.p_states: [observation]})[0]
        return np.random.choice(range(self.n_actions), p = p)

    def episode_start(self):
        """ Called each time a new episode starts """
        pass

    def episode_end(self):
        """ Called each time an episode ends """
        pass

    def process_feedback(self, state, action, reward, cost, state_new, done, info):
        """ Called inside the train loop, typically just stores the data """
        self.buffer.append((state, action, reward, cost, done))
        pass

    def train_start(self):
        """ Called before one training phase """
        self.buffer = []
        pass

    def train(self):
        """ Train method, typically runs on a GPU """
        # unpacking the data from the buffer...
        S, A, R, C, D = zip(*self.buffer)
        
        # calculating constraint return...
        constraint_return = estimate_constraint_return(C, D, self.gamma)

      #  print("Estimated return from %s %s: %d" % (str(C), str(D), constraint_return))

        # creating a feed dict for TF
        feed_dict = {self.p_states: S, self.p_actions: A, self.p_rewards: R,
            self.p_discounted_rewards_to_go: discount_many(R, D, self.gamma), self.p_disc_costs: discount_many(C, D, self.gamma),
            self.p_constraint_return: constraint_return}

        for i in range(self.steps):            
            # running the train op
            result = self.sess.run([self.op_step] + self.metrics, feed_dict = feed_dict)
        
        # returning the result (all but step)
        #assert all([is_number(r) for r in result[1:]]), "Only support scalar metrics, but got " + str(result[1:])
        return {x.name: np.mean(y) for x, y in zip(self.metrics, result[1:])}

