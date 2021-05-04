from gym import Env, spaces
import pandas as pd
import numpy as np
import random

MAXIMUM_PERFORMANCE = 2000000
NUMBER_OF_ACTIONS = 3
NAME_OF_STABLE_COLUMN = ""
MAX_STEPS = 20000

THRESHOLD_FOR_2G = 12000
THRESHOLD_FOR_3G = 32000

INITIAL_SCORE = 1000


class RATEnv(Env):
    """ A RAT environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(RATEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAXIMUM_PERFORMANCE)

        # Actions of the format Choose 3g, choose 2g and choose LTE
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        self.current_step = random.randint(
            0, len(self.df.loc[:, NAME_OF_STABLE_COLUMN].values) - 6)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, NAME_OF_STABLE_COLUMN].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.score * delay_modifier

        done =

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = random.randint(
            0, len(self.df.loc[:, NAME_OF_STABLE_COLUMN].values) - 6)
        self.score = INITIAL_SCORE

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

    def _next_observation(self):
        # Go through the
        return self.df.sample()

    def _take_action(self, action):
        # take an action
        callType = action["service_type"]
#         If(calltype=Cs)

#         If(Lg < Tg)
#            If(Term=Class A)
#                Allocate Service to 2G
#             End

#         ElseIf(Lu < Tu)

#            If (Term  = Class B)
#                Allocate Service to 3G
#             End

#         End


# ElseIf(calltype= Is)

#    If (Lu < Tu)
#        If (Term  = Class B)
#            Allocate Service to 3G
#         End

#     ElseIf(Lw < Tw)

#        If (Term  = Class A)
#            Allocate Service to WLAN
#         End

#     ElseIf(Le < Te)

#        If (Term  = Class C)
#            Allocate Service to LTE
#         End
#     End

# ElseIf(calltype= Bs)

#    If (Lu < Tu)
#        If (Term  = Class B)
#            Allocate Service to 3G
#         End

#     ElseIf(Lw < Tw)

#        If (Term  = Class A)
#            Allocate Service to WLAN
#         End

#     ElseIf(Le < Te)

#        If (Term  = Class C)
#            Allocate Service to LTE
#         End
#     End

# ElseIf(calltype= Ss)

#    If (Le < Te)
#        If (Term  = Class C)
#            Allocate Service to LTE
#         End

#     ElseIf(Lu < Tu)

#        If (Term  = Class B)
#            Allocate Service to 3G
#         End

#     ElseIf(Lw < Tw)

#        If (Term  = Class A)
#            Allocate Service to WLAN
#         End
#     End

# End

    pass
