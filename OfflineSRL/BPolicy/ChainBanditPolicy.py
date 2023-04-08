"""
Select action -1 with probability third_action_prob.
Select action 0 with probability (1-third_action_prob)/2.
Select action 1 with probability (1-third_action_prob)/2.
"""

import random

class ChainBanditPolicy():

    def __init__(self, chainbanditmdp, third_action_prob = 0.2):
        self.num_states = chainbanditmdp.num_states
        self.actions = chainbanditmdp.ACTIONS
        self.third_action_prob = third_action_prob

    def _get_probs(self, state, timestep = 0):
        #return [(1-self.third_action_prob)/2, (1-self.third_action_prob)/2, self.third_action_prob]
        return [self.third_action_prob, (1 - self.third_action_prob) / 2, (1 - self.third_action_prob) / 2]

    def _get_action_prob(self, state, action, timestep = 0):
        index = self.actions.index(action)
        probs = self._get_probs(state,timestep)
        return probs[index]

    def _get_action(self, state, timestep = 0):
        probs = self._get_probs(state,timestep)
        action = random.choices(self.actions, weights=probs)[0]
        return action

        #randnum = random.uniform(0,1)
        #if randnum < self.third_action_prob:
        #    return self.actions[0]
        #else:
        #    return random.choice(self.actions[1:])