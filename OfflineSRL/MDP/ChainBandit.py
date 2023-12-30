'''
We describe the ChainBandit environment

Actions: [Fixed]
- Actions [-1,0,1].

States: [Fixed]
- Is a list of two integers. (Format: [.,.])
- First component is a number between 1 and num_states.
- Second component is a flag denoting bandit or MDP environment.
- The flag is 0 for the MDP environment and 1 for bandit environments.
- There is an additional absorbing state denoted by [-1,-1]

Initial state:
- With probability self.bandit_prob: [1,1]
- else: [1,0]

Reward model (Bandit environment, second component 1):
- For action = -1:
    - reward is always zero.
- For action = 0:
    - At even states (first component is even), expected reward is 1/(2*num_states).
    - At odd states, reward is always 0.
- For action = 1:
    - At odd states (first component is odd), expected reward is 1/(2*num_states).
    - At even states, reward is always 0.

Reward model (MDP environment):
- For states [x,0] where x<num_states:
    - For action = -1:
        - reward is always 1/2.
    - For action = 0:
        - At even states (x is even), expected reward is 1/(2*num_states).
        - At odd states, reward is always 0.
    - For action = 1:
        - At odd states (x is odd), expected reward is 1/(2*num_states).
        - At even states, reward is always 0.
- At state [num_states,0].
    - Reward is 1 for both actions [0,1].
    - Reward is 1/2 for action -1.

Transition model (Bandit environment, second component 1):
- Transitions are same for all actions.
    - [x,1] -> [x+1,1] for all x<num_state.
    - [num_state,1] -> [-1,-1].

Transition model (MDP environment, second component 0):
- For all actions:
    - [num_state,0] -> [-1,-1].
- For actions [0,1]:
    - [x,0] -> [x+1,0].
- For action -1:
    - [x,0] -> [x+1,1]
'''

# Python imports.
from __future__ import print_function

import copy
import random

# Other imports.
from OfflineSRL.MDP.old_MDP import MDP
from OfflineSRL.MDP.old_State import State
from OfflineSRL.MDP.ChainBanditState import ChainBanditState


class ChainBanditMDP(MDP):
    ''' Implementation for a standard Chain MDP '''

    ACTIONS = [[-1],[0],[1]]

    def __init__(self, num_states=5, gamma=1, bandit_prob = 0.5, bandit_rew_scale = 0.1, mdp_to_bandit_reward = 0.5):
        '''
        Args:
            num_states (int) [optional]: Number of states in the chain.
        '''
        # Add initial randomization
        self.bandit_prob = bandit_prob
        self.num_states = num_states
        self.bandit_rew_scale = bandit_rew_scale
        self.mdp_to_bandit_reward = mdp_to_bandit_reward
        MDP.__init__(self, ChainBanditMDP.ACTIONS, self._transition_func, self._reward_func, init_state=self._get_initial_state(),
                     gamma=gamma)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["num_states"] = self.num_states

        return param_dict

    def _get_initial_state(self):
        randint = random.uniform(0,1)
        is_bandit = randint < self.bandit_prob
        return ChainBanditState([1,0])

    def _reward_func(self, state, action, next_state=None):
        '''
        Args:
            state (State)
            action (str)
            next_state (State)

        Returns
            (float)
        '''
        # MDP environment reward
        if state.num_list[1]==0:
            if action[0] == -1:
                return self.mdp_to_bandit_reward
            # For actions [0,1].
            elif action[0] in [0,1]:
                # Special reward for last state transition.
                if state.num_list[0] == self.num_states:
                    return 1
                elif state.num_list[0]%2 == action:
                    randint = random.uniform(0, 1)
                    return randint < self.bandit_rew_scale/(self.num_states)
                else:
                    return 0

        # Bandit environment reward
        elif state.num_list[1]==1:
            # For action [0,1].
            if action[0] in [0,1]:
                if state.num_list[0]%2 == action[0]:
                    randint = random.uniform(0, 1)
                    return randint < self.bandit_rew_scale / (self.num_states)
                else:
                    return 0
            # For action -1.
            else:
                return 0

        # Absorbing state
        elif state.num_list[1] == -1:
            return 0

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        state = copy.deepcopy(state)
        # We are in a bandit environment.
        if state.num_list[1]==1:
            if state.num_list[0] == self.num_states:
                state.num_list = [-1,-1]
            else:
                state = state + 1
        # We are in an MDP environment.
        elif state.num_list[1]==0:
            if state.num_list[0] == self.num_states:
                state.num_list = [-1,-1]
            elif action[0] in [0,1]:
                state = state + 1
            # else state.num_list[0]<self.num_states and action = -1
            else:
                # Transition into bandit environment.
                state.num_list[1]=1
                state = state + 1
                #state = state + (-1)
        # Absorbing state
        elif state.num_list[1]==-1:
            # No change.
            state = state

        return state

    def __str__(self):
        return "chain-" + str(self.num_states)

    def reset(self):
        self.cur_state = self._get_initial_state()
