from saferl import *
import tensorflow_probability as tfp
import cvxpy as cp
from tf_helpers import *
import numpy as np

class ConstrainedRandomAgent(ConstrainedAgent):
    """ An RL agent for CMDPs which does random action choices """
    def __init__(self, env, *args, **kwargs):
        """ Initialize for environment """
        super(ConstrainedRandomAgent, self).__init__(env)

    def sample_action(self, observation):
        """ Sample an action given observation, typically runs on a GPU """
        assert observation is not None
        return np.random.choice(self.n_actions)

    def episode_start(self):
        """ Called each time a new episode starts """
        pass

    def episode_end(self):
        """ Called each time an episode ends """
        pass

    def process_feedback(self, state, action, reward, cost, state_new, done, info):
        """ Called inside the train loop, typically just stores the data """
        pass

    def train_start(self):
        """ Called before one training phase """
        pass

    def train(self):
        """ Train method, typically runs on a GPU """
        return {'foo': 'bar'}
        pass

class ConstrainedPolicyOptimization(ConstrainedAgent):
    """ CPO agent """
    def __init__(self, env, sess, delta = 0.01):
        """ Initialize for environment """
        super(ConstrainedPolicyOptimization, self).__init__(env)
        self.delta = delta

        self.sess = sess

        # number of dimensions of state space
        S_DIM = self.state_dim

        # number of available actions
        ACTIONS = self.n_actions

        # states
        self.states = tf.placeholder(tf.float64, shape=(None, S_DIM,))

        # taken actions
        self.actions = tf.placeholder(tf.int64, shape=(None,))

        # rewards obtained
        self.disc_rewards = tf.placeholder(tf.float64, shape=(None,))

        # costs obtained
        self.disc_costs = tf.placeholder(tf.float64, shape=(None,))

        # layers
        with tf.name_scope('layers'):
            # creating a model with one variable parameters
            model = FCModelConcat([S_DIM, 10, ACTIONS])

            # output of the model
            output = model.forward(self.states)

            # softmax to make probability distribution
            self.logits = tf.nn.softmax(output)

            # predicted labels
            self.labels = tf.argmax(self.logits, axis=1)

            # parameters = 1 variable ONLY
            self.params = model.W

        # one-hot encoded actions
        self.a_one_hot = tf.one_hot(self.actions, ACTIONS)

        # taken logits
        # logits_taken = tf.gather(logits, actions, axis = 1)
        self.logits_taken = tf.boolean_mask(self.logits, self.a_one_hot)

        # logarithm
        self.log_logits = tf.log(self.logits_taken)

        # calculated loss
        self.loss_r = -tf.reduce_mean(tf.multiply(self.log_logits, self.disc_rewards))
        self.loss_c = -tf.reduce_mean(tf.multiply(self.log_logits, self.disc_costs))

        # KL(pi(a)||pi(a) fixed)
        self.kl_div_var_fixed = tf.reduce_mean(tfp.distributions.kl_divergence(
            tf.distributions.Categorical(probs=self.logits),
            tf.distributions.Categorical(probs=tf.stop_gradient(self.logits))))
        # tf.distributions.Categorical(probs = logits))) #!!! not using this because want variable first parameter and fixed second

        # policy gradient for reward
        self.g = tf.gradients(-self.loss_r, self.params)[0]

        # policy gradient for constraint
        self.b = tf.gradients(-self.loss_c, self.params)[0]

        # hessian of KL divergence (parameter H)
        self.H = tf.hessians(self.kl_div_var_fixed, self.params)[0]

        # buffer for rollouts
        self.buffer = []

    def sample_action(self, observation):
        """ Sample an action given observation, typically runs on a GPU """
        p = self.sess.run(self.logits, feed_dict={self.states: [observation]})[0]
        return np.random.choice(range(self.n_actions), p=p)

    def episode_start(self):
        """ Called each time a new episode starts """
#        self.buffer.append([])
        pass

    def episode_end(self):
        """ Called each time an episode ends """
        pass

    def process_feedback(self, state, action, reward, cost, state_new, done, info):
        """ Called inside the train loop, typically just stores the data """
        self.buffer.append((state, action, reward, cost, done))

    def train_start(self):
        """ Called before one training phase """
        self.buffer = []

    def train(self):
        """ Train method, typically runs on a GPU """

        ### TODO: reimplement w/o inner optimization, otherwise too slow.

        ## TODO: see how norm /delta evolves!
        ## TODO: implement backtracking line search

        def solve_CPO(H_, b_, g_, J_R, J_C, delta = 0.1, d = self.threshold, H_lambda=0.01):
            def load_params(vector):
                """ Load params from vector """
                self.sess.run(self.params.assign(vector))


            # CPO hyperparameters

            # D_KL maximal distance
            # delta

            # maximal constraint return
            # d

            # matrix conditioner
            # H_lambda

            # calculating H, g, b, J from obtained data
            J_c_ = J_C

            # conditioning
            H_ = H_ + H_lambda * np.eye(H_.shape[0])

            # constraint dissatisfaction
            c = J_c_ - d

            # current parameters
            theta_k = self.sess.run(self.params)


            # Construct the problem.
            theta = cp.Variable(len(theta_k))
            objective = cp.Maximize(g_.T @ (theta - theta_k))
            constraints = [c + b_.T @ (theta - theta_k) <= 0, cp.quad_form(theta - theta_k, H_) <= 2 * delta]
            prob = cp.Problem(objective, constraints)

            # The optimal objective value is returned by `prob.solve()`.
            fallback = False
            try:
                result = prob.solve(gp=False)
            except:
                fallback = True

            if fallback or theta.value is None:
#                print(J_c_, d)
                print('Optimization failed: current policy nonsafe. Falling back to a safe solution...')

                # θ∗ = θk −sqrt(2δ/b^T H−1b)H^-1b. (eq 14)
                Hinv = np.linalg.inv(H_)
                theta_safe = theta_k - 0.5 * ((2 * delta / (b_.T @ Hinv @ b_)) ** 0.5 * Hinv @ b_).reshape(-1)
                load_params(theta_safe)
                return

            load_params(theta.value)

        def get_HbgRC(rollouts):
            # unpacking rollouts
            S, A, R, C, D = zip(*rollouts)

            #global summary_i, s_writer
            H_, b_, g_ = self.sess.run([self.H, self.b, self.g],
                                               feed_dict={self.states: S, self.actions: A, self.disc_rewards: discount_many(R, D, self.gamma),
                                                          self.disc_costs: discount_many(C, D, self.gamma)})

            # discounting over many episodes
            J_C = estimate_constraint_return(C, D, self.gamma)
            J_R = estimate_constraint_return(R, D, self.gamma)

            return np.array(H_), np.array([b_]).T, np.array([g_]).T, J_R, J_C

        H_, b_, g_, J_R, J_C = get_HbgRC(self.buffer)
        solve_CPO(H_, b_, g_, J_R, J_C, delta = self.delta, d = self.threshold, H_lambda = 0.01)
        return {}
