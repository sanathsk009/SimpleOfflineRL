"""
Randomly sample actions in mdp.
"""

import random

class UniformRandomPolicy():

    def __init__(self, mdp):
        self.actions = mdp.ACTIONS

    def _get_probs(self, state, timestep = 0):
        return [1/len(self.actions)]*len(self.actions)

    def _get_action_prob(self, state, action, timestep = 0):
        index = self.actions.index(action)
        probs = self._get_probs(state,timestep)
        return probs[index]

    def _get_action(self, state, timestep = 0):
        probs = self._get_probs(state,timestep)
        action = random.choices(self.actions, weights=probs)[0]
        return action