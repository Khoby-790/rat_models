from gym import Env, spaces
import pandas as pd
import numpy as np

MAXIMUM_PERFORMANCE = 2000000
NUMBER_OF_ACTIONS = 3


class RATEnv(Env):
    """ A RAT environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(RATEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAXIMUM_PERFORMANCE)

        # Actions of the format Choose 3g, choose 2g and choose LTE
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        self.curr

    def step(self, action):
        # Execute one time step within the environment
        ...

    def reset(self):
        # Reset the state of the environment to an initial state
        ...

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

    def _next_observation(self):
        # Go through the
        pass

    def _take_action(self, action):
        # take an action
        pass
