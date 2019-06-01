from sppo import *
from baselines import *
from tf_helpers import *
from saferl import *

def test_random():
    env = make_safe_env('CartPole-v0-left-half')
    agent = ConstrainedRandomAgent(env)
    loop = ConstrainedEpisodicTrainLoop(env, agent)
    loop.rollout()
    assert loop.train_step() == True, "Random agent must return True on training"

def test_sppo():
    env = make_safe_env('CartPole-v0-left-half')
    sess = create_modest_session()
    agent = ConstrainedProximalPolicyOptimization(env, sess)
    loop = ConstrainedEpisodicTrainLoop(env, agent)
    sess.run(tf.global_variables_initializer())
    loop.rollout()
    loop.train_step()
    assert True