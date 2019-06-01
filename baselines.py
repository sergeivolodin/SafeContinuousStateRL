from saferl import *

class ConstrainedRandomAgent(ConstrainedAgent):
    """ An RL agent for CMDPs which does random action choices """
    def __init__(self, env):
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
        return True
        pass

