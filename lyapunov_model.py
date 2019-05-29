import gym
from time import sleep
import pulp
import tensorflow as tf
from tqdm import tqdm
from lyapunov_helpers import *
import datetime

tf.reset_default_graph()
# allowing GPU memory growth to allocate only what we need
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config, graph = tf.get_default_graph())

# number of dimensions in state
S_DIM = 4

# number of available actions
ACTIONS = 2

# discount factor
gamma_discount = 0.9

# epsilon-greedy
eps = 0.1

# for softmax
temperature = 1

# maximal constraint violation
d0 = tf.placeholder(tf.float64)

# states: double the length of actions: from-to-from-to...
states = tf.placeholder(tf.float64, shape = (None, S_DIM,))

# taken actions
actions = tf.placeholder(tf.int64, shape = (None,))

# rewards obtained
rewards = tf.placeholder(tf.float64, shape = (None,))

# costs obtained
costs = tf.placeholder(tf.float64, shape = (None, ))

# is step terminal?
termination = tf.placeholder(tf.float64, shape = (None, ))

# need next q-value (non-terminal state?)
need_next_q = tf.placeholder(tf.float64, shape = (None, ))

# new policy pi'
logits_new_policy = tf.placeholder(tf.float64, shape = (None, ACTIONS))

# one-hot encoded actions
a_one_hot = tf.one_hot(actions, ACTIONS)

common_layer = states#fc_layer(states, 20)

q_reward, _ = q_like_function(common_layer, ACTIONS = ACTIONS, temperature = temperature, name = 'reward')
q_termination, _ = q_like_function(common_layer, ACTIONS = ACTIONS, temperature = temperature, name = 'termination')
q_cost, _ = q_like_function(common_layer, ACTIONS = ACTIONS, temperature = temperature, name = 'cost')
_, logits_policy = q_like_function(common_layer, ACTIONS = ACTIONS, temperature = temperature, name = 'policy')

# taken logits (from state)
logits_from = logits_policy[0::2]

with tf.name_scope('q_lyapunov'):
    eps_lyapunov = d0 - dot_and_sum1(logits_policy, q_cost) / dot_and_sum1(logits_policy, q_termination)
    q_lyapunov = q_cost + tf.multiply(q_termination, tf.expand_dims(eps_lyapunov, 1))

# Bellmann losses
#loss_r = soft_bellman_loss(a_one_hot, rewards, q_reward, logits_new_policy, need_next_q)
with tf.name_scope('loss_reward'):
    loss_r = soft_bellman_loss(a_one_hot, rewards, q_reward, logits_new_policy, need_next_q, gamma_discount = gamma_discount)
#loss_r = soft_bellman_loss(a_one_hot, rewards, q_reward, logits_from, need_next_q)
#loss_r = soft_bellman_loss(a_one_hot, rewards, q_reward, logits_reward[0::2], need_next_q)
with tf.name_scope('loss_cost'):
    loss_d = soft_bellman_loss(a_one_hot, costs, q_cost, logits_from, need_next_q, gamma_discount = gamma_discount)
with tf.name_scope('loss_termination'):
    loss_t = soft_bellman_loss(a_one_hot, termination, q_termination, logits_from, need_next_q, gamma_discount = gamma_discount)

# logits of policy (taken, from original state)
# using MSE instead of JSD...
#loss_jsd = tf.reduce_mean(mse(logits_from, logits_new_policy))
with tf.name_scope('loss_jsd'):
    loss_jsd = tf.reduce_mean(jsd(logits_from, logits_new_policy))
    
# using a separate optimizer for each loss (can have different coefficients...)
losses = [loss_r, loss_d, loss_t, loss_jsd]
bellman_losses = [loss_r, loss_d, loss_t]
    

# iteration for Bellman updates
opt_bellman = loss_to_opt(bellman_losses, name = 'bellman_opt')
opt_jsd = loss_to_opt([loss_jsd], name = 'jsd_opt')

tf.summary.scalar('q_norm_r', tf.reduce_mean(tf.norm(q_reward, axis = 1)))
tf.summary.scalar('q_norm_d', tf.reduce_mean(tf.norm(q_cost, axis = 1)))
tf.summary.scalar('q_norm_t', tf.reduce_mean(tf.norm(q_termination, axis = 1)))
tf.summary.scalar('loss_r', loss_r)
tf.summary.scalar('loss_d', loss_d)
tf.summary.scalar('loss_t', loss_t)
tf.summary.scalar('loss_jsd', loss_jsd)
tf.summary.scalar('mean_act', tf.reduce_mean(tf.cast(tf.argmax(a_one_hot, axis = 1), dtype=tf.float64)))
tf.summary.scalar('episode_length', tf.reduce_sum(tf.shape(rewards)))
tf.summary.scalar('reward', tf.reduce_sum(rewards))
tf.summary.scalar('termination', tf.reduce_sum(termination))
tf.summary.scalar('cost', tf.reduce_sum(costs))
tf.summary.scalar('pi_vs_qR', mse(tf.argmax(q_reward[0::2], axis = 1), tf.argmax(logits_from, axis = 1)))

summary = tf.summary.merge_all()
checkpoint_i = 0
summary_i = 0
summary_name = str(datetime.datetime.now())
s_writer = tf.summary.FileWriter('./tensorboard/' + summary_name)

env = gym.make('CartPole-v0')
print('Creating the environment')

def sample_action(observation, sample_q = False, eps = 0):
    """ Sample an action from the policy """
    
    if sample_q: # sampling from the Q-function
        if np.random.random() <= eps:
            return env.action_space.sample()
        
        p = sess.run(q_reward, feed_dict = {states: [observation]})[0]
        # argmax for q-function
        return np.argmax(p)
    else: # sampling from the policy
        p = sess.run(logits_policy, feed_dict = {states: [observation]})[0]
    
        # choice for real policy
        return np.random.choice(range(2), p = p)

def get_rollout(do_render = False, delay = None, **kwargs):
    """ Obtain rollout using policy """
    done = False
    observation = env.reset()
    sarsnc = []
    # S: from state
    # A: action
    # R: reward obtained
    # S: new state
    # N: true if need next (non-terminal)
    # C: cost obtained
    if do_render: env.render()
    while not done:
        act = sample_action(observation, **kwargs)
        observation_, reward, done, info = env.step(act) # take a random action
        if do_render: env.render()
        curr = (observation, act, reward, observation_, True, cost(observation_))
        sarsnc.append(curr)
        replay.store(curr)
        
        if done:
            # crucial, otherwise Q blows up...
            replay.store((observation_, sample_action(observation_), reward, observation_, False, cost(observation_)))

        if delay: sleep(delay)
        
        observation = observation_
    env.close()
    #print(len(replay.buf), len(sars))
    return sarsnc

def train_step(L = 200, d0_ = 50, learn_policy = False, learn_q = False, do_safe = False, only_constraint = False):
    """
    L: size to sample from the replay buffer
    d0_: constraint return threshold
    learn_policy: Do a JSD update on the policy (supervised step)
    learn_q: do a Bellman update for q functions
    do_safe: include safety Lyapunov constraints into the LP
    only_constraint: only care about the constraint minimization, w/o optimizing for reward (for initial stage)
    """
    
    # sampling same size
    S0, A, R, S1, N, C = list(zip(*replay.sample(L)))
    
    # taking some data from the environment (sampling...)
    #S0, A, R, S1, N, C = zip(*get_rollout())

    # converting to float
    N = 1. * np.array(N)

    # feed dictionary for TF
    feed = {states: interleave(S0, S1), actions: A, rewards: R, costs: C,
            d0: d0_, need_next_q: N, termination: 1 - N}
    
    # obtaining current policy
    pi_k, q_L, q_C, q_R, eps_L = sess.run((logits_from, q_lyapunov, q_cost, q_reward, eps_lyapunov), feed_dict = feed)
    
    #print(eps_L, q_L, q_R)
    
    # taking FROM for q_L, q_R, eps_L
    q_L = q_L[0::2,:]
    q_C = q_C[0::2,:]
    q_R = q_R[0::2,:]
    eps_L = eps_L[0::2]
    
    #print(q_L)
    
    # sanity check for length
    assert len(q_L) == len(q_R) == len(eps_L) == len(q_C) == L

    # if item is 0, replace it to this
    # to fix KL divergence blowing up
    kl_eps = 1e-2
    
    # lower/upper bound for variables
    lb, ub = kl_eps, 1.0
    
    # number of variables: |A| * |Batch|
    n_vars = (pi_k.shape[0] * ACTIONS)
    
    # new policy is initialy None
    pi_new = None
    
    # solving the LP
    # Use https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    # to solve a linear program
    err_info = None
    try:
        # creating a maximization problem
        prob = pulp.LpProblem("policy", pulp.LpMaximize)
        
        # creating variables lb <= x_i <= ub
        x = pulp.LpVariable.dicts('x', range(n_vars), lowBound=lb, upBound = ub)
        
        # flattening q_R S1A1 S1A2 S2A1 S2A2 ...
        # OR -q_L
        q_target = -q_C if only_constraint else q_R
        q_flatten = q_target.flatten()
        
        # all variables flattened
        x_vars = [x[i] for i in range(n_vars)]
        
        # sanity check for length of q-function
        assert len(x_vars) == len(q_flatten)
        
        # ONLY optimize for the constraint (minimize)
        # adding objective: (Q_R, x)
        prob += pulp.lpSum([Qi * xi for Qi, xi in zip(q_flatten, x_vars)]), "objective"

        # adding constraints
        for i in range(n_vars // ACTIONS):
            # adding probability constraint
            prob += pulp.lpSum([x[2 * i], x[2 * i + 1]]) == 1.0

            ## NOT including safety...
            #continue
            if do_safe:
                assert not only_constraint, "Trying to enable BOTH constraints and optimizing for low cost"
                cqL = q_L[i, 0]
                cqR = q_L[i, 1]
                xL = x[2 * i]
                xR = x[2 * i + 1]
                #print('Constraint', eps_L[i], cqL, cqR)
                prob += pulp.lpSum([xL * cqL, xR * cqR]) <= eps_L[i] + pi_k[i,0] * cqL + pi_k[i,1] * cqR

        # solving the problem
        status = prob.solve()

        # obtaining new policy
        pi_new = np.array([xx.value() for xx in x_vars])
        #print(pulp.LpStatus[status])
        if pulp.LpStatus[status] != 'Optimal':
            #print('NotOptimal')
            raise Exception('Problem was not solved ' + pulp.LpStatus[status])
    except:
        err_info = sys.exc_info()
        pass
        # otherwise, it's N
        pi_new = float('nan')
        
    # if no new policy, assign it to old policy
    #print(q_R)
    if pi_new is None or np.isnan(np.linalg.norm(pi_new)):
        print('Optimization for pi failed... Only doing Bellman update', err_info)
        pi_new = pi_k
    else:
        
        
        #print('iszero', pi_new, pi_new == 0.0)

        # fixing kl_eps
        #pi_new[pi_new == 0.0] = kl_eps

        # next policy (zipping back)
        pi_new = np.array([pi_new[0::2], pi_new[1::2]]).T
    
    # argmax(q) = argmax(pi_new)
    #assert np.allclose(np.argmax(q_R, axis = 1), np.argmax(pi_new, axis = 1))

    # adding data to FEED dict
    feed[logits_new_policy] = pi_new
    
   # print(pi_new)
    
    if learn_q:
        # running BELLMAN update
        summary_res, _ = sess.run([summary, opt_bellman], feed_dict = feed)
        global summary_i, s_writer
        s_writer.add_summary(summary_res, summary_i)
        summary_i += 1
    
    #print(pi_k, pi_new)
    #print("pi_k avg, pi_new avg", np.mean(np.argmax(pi_k, axis = 1)), np.mean(np.argmax(pi_new, axis = 1)), np.min(pi_k), np.max(pi_k))
    
    # only learn policy if said to do so
    if learn_policy:
        # training 10 times for JSD
        for i in range(1):
            summary_res, _ = sess.run([summary, opt_jsd], feed_dict = feed)
            s_writer.add_summary(summary_res, summary_i)
            summary_i += 1
    
    if False:
        print('pi_k', pi_k)
        print('q_L', q_L)
        print('q_R', q_R)
        print('eps_L', eps_L)
        print('pi_new', pi_new)
        print('l_j', l_j)
    
    return np.sum(R), np.sum(C)

def init_agent():
    """ Initialize weights (reset) """
    init = tf.global_variables_initializer()
    sess.run(init)
    global replay
    replay = ExperienceReplay(N = 5000)

def learning_iterations(iterations, eps_decay, rollouts_per_train, L = 100, d0_ = 1000, learn_policy = True,
                        learn_q = True, do_safe = False, only_constraint = True, sample_q = False):
    """ Learn with parameters """
    
    print('Parameters: ' + str(locals()))
    plot_decay(eps_decay)
    
    r, c = [], []
    for i in tqdm(range(iterations)):
        # obtaining current epsilon for exploration
        eps = get_current_eps(eps_decay, i)

        # obtaining rollout
        for i in range(rollouts_per_train):
            sarsnc = get_rollout(eps = eps, sample_q = sample_q)
            _, _, R, _, _, C = zip(*sarsnc)
            r.append(np.sum(R))
            c.append(np.sum(C))

        # training the agent
        train_step(L = L, d0_ = d0_, learn_policy = learn_policy, learn_q = learn_q, do_safe = do_safe,
                   only_constraint = only_constraint)
    plot_performance(r, c)
    print_info(get_rollout())

def enable_video():
  global env
  print('Enabling video')
  env = gym.wrappers.Monitor(env, "./gym-results/", force = True)

def disable_video():
  global env
  env = env.unwrapped

def restore(fn):
  tf.train.Saver().restore(sess, fn)

def checkpoint():
    """ Save data """
    global checkpoint_i
    fn = './' + summary_name + "_" + str(checkpoint_i) + ".ckpt"
    checkpoint_i += 1
    tf.train.Saver().save(sess, fn)
    print('Saved to %s' % fn)

def discount(rewards):
    """ Discount and do cumulative sum """
    sum_so_far = 0.0
    rewards_so_far = []
    for r in rewards[::-1]:
        sum_so_far = sum_so_far * gamma_discount + r
        rewards_so_far.append(sum_so_far)
    return rewards_so_far[::-1]

assert np.allclose(discount([1,2,3]), [3 * gamma_discount ** 2 + 2 * gamma_discount + 1, 3 * gamma_discount + 2, 3]), "Check discount implementation"

def print_info(rollout):
    _,a_,r_,_,_,c_ = zip(*rollout)
    print('Mean action: %.2f' % np.mean(a_))
    print('Total reward: %d / disc %.2f' % (np.sum(r_), discount(r_)[0]))
    print('Total cost: %d / disc %.2f' % (np.sum(c_), discount(c_)[0]))

