from tqdm import tqdm
from IPython.display import clear_output
import tensorflow as tf
from matplotlib import pyplot as plt
import gym
from costs import *

class ConstrainedEnvironment():
    """ Wrapper around gym environment with cost/threshold """
    def __init__(self, env, cost, threshold, gamma = 1.0):
        """ Initialize a constrained environment with discount gamma """
        assert hasattr(env, 'spec') and hasattr(env.spec, '_env_name'), "Please supply a valid Gym environment"
        assert hasattr(cost, '__call__'), "Cost must be a function state -> cost (real)"
        assert isinstance(threshold, float), "Threshold must be a real-valued number"

        self.env = env
        self.cost = cost
        self.threshold = threshold
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.gamma = gamma

    def reset(self):
        """ Reset the environment """
        return self.env.reset()

    def render(self):
        """ Render current state """
        return self.env.render()

    def step(self, action):
        """ Take an action and return obs, rew, cost, done, info """
        obs, rew, done, info = self.env.step(action)
        return (obs, rew, self.cost(obs), done, info)

    def close(self):
        """ Close the gym environment """
        return self.env.close()

class ConstrainedAgent():
    """ An RL agent for constrained MDPs with discrete actions and continuous states """
    def __init__(self, env):
        """ Initialize for environment """
        assert isinstance(env.action_space, gym.spaces.discrete.Discrete), "Only support discrete actions """
        assert len(env.observation_space.shape) == 1, "Only support 1D actions """
        self.state_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.gamma = env.gamma
        self.threshold = env.threshold

        # list of metrics to track
        self.metrics = []

    def sample_action(self, observation):
        """ Sample an action given observation, typically runs on a GPU """
        raise NotImplementedError("Trying to call an abstract class method")

    def episode_start(self):
        """ Called each time a new episode starts """
        raise NotImplementedError("Trying to call an abstract class method")

    def episode_end(self):
        """ Called each time an episode ends """
        raise NotImplementedError("Trying to call an abstract class method")

    def process_feedback(self, state, action, reward, cost, state_new, done, info):
        """ Called inside the train loop, typically just stores the data """
        raise NotImplementedError("Trying to call an abstract class method")

    def train_start(self):
        """ Called before one training phase """
        raise NotImplementedError("Trying to call an abstract class method")

    def train(self):
        """ Train method, typically runs on a GPU. Returns a dict with summary """
        raise NotImplementedError("Trying to call an abstract class method")

    def track_metrics(self, lst):
        """ Track all metrics from the list """
        assert isinstance(lst, list), "Please provide a list"
        self.metrics = lst

class ConstrainedEpisodicTrainLoop():
    """ Loop agent-environment interaction for constrained env/agents """
    def __init__(self, env, agent, episodes_to_collect = 5):
        """" Initialize for env, agent and number episodes before calls to train() """
        self.env = env
        self.agent = agent
        self.episodes_to_collect = episodes_to_collect

    def achieve_reward(self, R_thresh, max_epochs, do_plot = False):
        """ Train until a reward is achieved """
        Rs, Cs = [], []
        for i in tqdm(range(max_epochs)):
            # training
            rollouts, _ = self.train_step()

            # obtaining mean reward/constraint over batches
            R = np.mean([np.sum(r[1]) for r in rollouts])
            C = np.mean([np.sum(r[2]) for r in rollouts])

            Rs.append(R)
            Cs.append(C)

            # plotting if requested
            if i > 0 and i % 100 == 0 and do_plot:
                plot_RC(Rs, Cs, self.env.threshold)

            # stopping on success
            if R >= R_thresh:
                break

        # plotting for the last time
        if do_plot: plot_RC(Rs, Cs, self.env.threshold)

        # return rewards/constraints
        return Rs, Cs

    def train_step(self):
        """ Train once """

        # informing the agent about incoming data
        self.agent.train_start()

        # data for rollouts
        rollouts = []

        # doing the interaction
        for i in range(self.episodes_to_collect):
            rollouts.append(self.rollout())

        # training the agent
        return rollouts, self.agent.train()

    def rollout(self):
        """ Run agent-environment interaction once """
        obs = self.env.reset()
        self.agent.episode_start()
        done = False

        # gathered rewards and constraints
        R, C = [], []
        L = 0 # episode length

        while not done:
            # agent takes an action
            act = self.agent.sample_action(obs)

            # the environment responds
            obs_, rew, cost, done, info = self.env.step(act)

            # increasing length
            L += 1
            R.append(rew)
            C.append(cost)

            # informing the agent
            self.agent.process_feedback(obs, act, rew, cost, obs_, done, info)

            # if finished, informing the agent
            if done: self.agent.episode_end()

            # updating the state
            obs = obs_

        return L, R, C

def make_safe_env(env_name, **kwargs):
    """ Factory function to create safe environments """
    # environment descriptions (gym env, cost fcn, threshold)
    envs = {'CartPole-v0-left-half': ['CartPole-v0', cost_cartpole_left, 100.0]}

    # only support cartpole now
    assert env_name in envs, "Only support " + str(envs.keys())

    # unpacking arguments
    env_name_gym, cost, threshold = envs[env_name]

    # creating an environment
    env_unsafe = gym.make(env_name_gym)

    return ConstrainedEnvironment(env_unsafe, cost, threshold, **kwargs)

def discount(rewards, gamma):
    """ Discount and do cumulative sum """
    sum_so_far = 0.0
    rewards_so_far = []
    for r in rewards[::-1]:
        sum_so_far = sum_so_far * gamma + r
        rewards_so_far.append(sum_so_far)
    return rewards_so_far[::-1]

def discount_many(r, d, gamma):
    """ Discount rewards from many episodes
        r: all rewards
        d: done (1 or 0)
    """
    assert len(r) == len(d), "Length of rewards must be the same as length of done"
    buffer = []
    result = []
    for r_, d_ in zip(r, d):
        buffer.append(r_)
        if d_: # if episode ended...
            result += discount(buffer, gamma)
            buffer = []
    return result
assert discount_many

assert discount_many(r = [1,1,1,1,1,1], gamma = 0.5, d = [0,0,0,1,0,1]) == [1.875, 1.75, 1.5, 1.0, 1.5, 1.0]

def plot_RC(Rs, Cs, threshold):
    """ Plot rewards/costs """
    clear_output()
    plt.plot(Rs, label = 'rewards', color = 'green')
    plt.plot(Cs, label = 'constraints', color = 'red')
    plt.axhline(y = threshold, ls = '--', color = 'red')
    plt.legend()
    plt.show()

def estimate_constraint_return(C, D, gamma):
    """ Estimate constraint return from constraints (array) and Done array """
    returns = []
    current = []
    for c, d in zip(C, D):
        current.append(c)
        if d:
            returns.append(discount(current, gamma)[0])
            current = []
    return np.mean(returns)
