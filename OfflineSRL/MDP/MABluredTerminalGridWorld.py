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

class MultiArmBluredTerminalGridWorldMDP(GridWorldMDP):
    ACTIONS = [["U"], ["D"], ["L"], ["R"]]

    def __init__(self,
                 width=5,
                 height=3,
                 ma = 2,
                 init_loc=(1, 1),
                 rand_init=False,
                 goal_locs=[()],
                 lava_locs=[()],
                 walls=[],
                 is_goal_terminal=True,
                 is_lava_terminal=False,
                 gamma=0.99,
                 slip_prob=0.0,
                 step_cost=0.0,
                 lava_cost=1.0,
                 name="gridworld"):

        GridWorldMDP.__init__(self, width=width,
                 height=height,
                 init_loc=init_loc,
                 rand_init=rand_init,
                 goal_locs=goal_locs,
                 lava_locs=lava_locs,
                 walls=walls,
                 is_goal_terminal=is_goal_terminal,
                 is_lava_terminal=is_lava_terminal,
                 gamma=gamma,
                 slip_prob=slip_prob,
                 step_cost=step_cost,
                 lava_cost=lava_cost,
                 name=name)

        self.ma = ma
        NEW_ACTIONS = []
        for action in self.ACTIONS:
            for iter in range(ma):
                NEW_ACTIONS.append([action[0]+str(1+iter)])
        self.ACTIONS = NEW_ACTIONS

    def _perturb(self, state):
        state_loc = state._vrepr()
        perturb_loc = 0
        if np.random.binomial(n=1,p=0.5):
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
        action_dic = {"L": "left", "R": "right", "U": "up", "D": "down"}
        grid_action = action_dic[action[0]]
        action_type = int(action[1:])

        cur_state = self.cur_state
        if (int(cur_state.x), int(cur_state.y)) in self.goal_locs or (int(cur_state.x), int(cur_state.y)) in self.lava_locs:
            # self._is_goal_state_action(state, action):
            self.cur_state = GridWorldState(0, 0)
            reward = 0
        elif cur_state.x == 0:
            reward = 0
        else:
            next_state = self.transition_func(self.cur_state, grid_action)
            next_state = self._perturb(next_state)
            reward = self.reward_func(self.cur_state, action, next_state)/action_type
            self.cur_state = next_state

        return reward, copy.deepcopy(self.cur_state)