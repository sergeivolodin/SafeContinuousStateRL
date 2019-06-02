from sppo import *
from baselines import *
from tf_helpers import *
from saferl import *

def test_random():
    """ Test that basic functionality works """
    env = make_safe_env('CartPole-v0-left-half')
    agent = ConstrainedRandomAgent(env)
    loop = ConstrainedEpisodicTrainLoop(env, agent)
    loop.rollout()
    assert isinstance(loop.train_step()[1], dict), "Random agent must return a dict on training"

def test_sppo():
    """ Test that sPPO can be created """
    env = make_safe_env('CartPole-v0-left-half')
    sess = create_modest_session()
    agent = ConstrainedProximalPolicyOptimization(env, sess)
    loop = ConstrainedEpisodicTrainLoop(env, agent)
    sess.run(tf.global_variables_initializer())
    loop.rollout()
    loop.train_step()
    assert True

def test_convergence_unsafe(R_thresh = 180, epochs = 10000):
    """ Test that PPO converges (threshold reached in fixed number of epochs) """
    env = make_safe_env('CartPole-v0-left-half')
    sess = create_modest_session()
    agent = ConstrainedProximalPolicyOptimization(env, sess, ignore_constraint = True, steps = 5)
    loop = ConstrainedEpisodicTrainLoop(env, agent, episodes_to_collect = 10)

    # RL algos can diverge after finding a solution
    # https://www.reddit.com/r/MachineLearning/comments/8pcykb/d_actor_critic_ddpg_diverging_after_finding/
    # Add KL-divergence check? (now missing...)

    sess.run(tf.global_variables_initializer())

    # train and hope...
    Rs, Cs = loop.achieve_reward(R_thresh = R_thresh, max_epochs = epochs, do_plot = True)    

    # check that the reward is indeed good
    assert Rs[-1] > R_thresh, "Must achieve %s in %d epochs, got only %d" % (R_thresh, epochs, Rs[-1])

test_sppo()
