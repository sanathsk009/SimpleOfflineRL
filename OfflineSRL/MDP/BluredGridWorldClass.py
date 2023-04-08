# Python imports.
from __future__ import print_function
import random
import sys, os
import copy
import numpy as np
from collections import defaultdict

# Other imports.
from OfflineSRL.MDP.old_MDP import MDP
from OfflineSRL.MDP.old_State import State
from OfflineSRL.MDP.GridWorldStateClass import GridWorldState
from OfflineSRL.MDP.GridWorldMDPClass import GridWorldMDP

class BluredGridWorldMDP(GridWorldMDP):
    ACTIONS = [["up"], ["down"], ["left"], ["right"], ["reset"]]

    def _perturb(self, state):
        state_loc = state._vrepr()
        perturb_loc = np.array([random.randint(-1, 1), random.randint(-1, 1)])
        new_loc = state_loc + perturb_loc
        new_loc[0] = max(new_loc[0], 1)
        new_loc[0] = min(new_loc[0], self.width)
        new_loc[1] = max(new_loc[1], 1)
        new_loc[1] = min(new_loc[1], self.height)
        while new_loc in self.walls:
            perturb_loc = np.array([random.randint(-1, 1), random.randint(-1, 1)])
            new_loc = state_loc + perturb_loc
            new_loc[0] = max(new_loc[0], 1)
            new_loc[0] = min(new_loc[0], self.width)
            new_loc[1] = max(new_loc[1], 1)
            new_loc[1] = min(new_loc[1], self.height)

        return GridWorldState(new_loc[0],new_loc[1])

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        '''
        action = action[0]
        if action == "reset":
            self.reset()
            reward = 0
        else:
            next_state = self.transition_func(self.cur_state, action)
            next_state = self._perturb(next_state)
            reward = self.reward_func(self.cur_state, action, next_state)
            self.cur_state = next_state

        return reward, copy.deepcopy(self.cur_state)