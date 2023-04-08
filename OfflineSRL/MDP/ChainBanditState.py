'''
Represents states as a list of two integers.
First element of list, denotes the state index.
Second element is a flag denoting bandit or MDP environment.
The flag is 0 for the MDP environment and 1 for bandit environments.
'''

# Python imports
import numpy as np

# Other imports.
from OfflineSRL.MDP.old_State import State

class ChainBanditState(State):
    ''' Class for Chain MDP States '''

    def __init__(self, num_list):
        State.__init__(self, data=num_list)
        self.num_list = np.array(num_list)

    def __hash__(self):
        return self.num_list

    def __add__(self, val):
        return ChainBanditState(self.num_list + np.array([val,0]))

    def __lt__(self, val):
        return self.num_list[0] < val

    def __str__(self):
        return "s." + str(self.num_list)

    def __eq__(self, other):
        '''
        Summary:
            Chain states are equal when their num is the same
        '''
        return isinstance(other, ChainBanditState) and self.num_list[0] == other.num_list[0] and self.num_list[1] == other.num_list[1]

    def _get_vector_rep(self):
        return self.num_list
