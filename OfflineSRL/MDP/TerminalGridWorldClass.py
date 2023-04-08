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

class TerminalGridWorldMDP(GridWorldMDP):

    def _go_terminal(self, next_state):
        cur_state = self.cur_state
        if (int(cur_state.x), int(cur_state.y)) in self.goal_locs or (int(cur_state.x), int(cur_state.y)) in self.lava_locs:
            # self._is_goal_state_action(state, action):
            return GridWorldState(0,0)
        else:
            return next_state

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
        next_state = self.transition_func(self.cur_state, action)
        next_state = self._go_terminal(next_state)
        if next_state.x == 0:
            reward = 0
        else:
            reward = self.reward_func(self.cur_state, action, next_state)
        self.cur_state = next_state

        return reward, copy.deepcopy(self.cur_state)