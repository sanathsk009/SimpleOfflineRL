"""
General points:
- All learners implemented here inherit from offlineTabularBase.
- These learners need to be initialized, and then fit on an MDPDataset.
- Most of these learners just implement pointers to the right agent via the set_agent method.

PesBandit:
- Inherits offlineTabularBase and sets agent to PesBanditAgent.

VI:
- Inherits offlineTabularBase and sets agent to VIAgent.

PVI:
- Inherits offlineTabularBase and sets agent to PVIAgent.
- Also modifies __init__ to modify self.scalar if needed when initialization, this gets passed down to PVIAgent.

SPVI:
- Inherits offlineTabularBase and sets agent to SPVIAgent.
- Also runs _extract_bpolicy which takes the original behavioral policy, and encodes the behavioral policy into a tabular format that is usable by the agents.
"""

from OfflineSRL.OfflineLearners.offlineBase import offlineTabularBase
from OfflineSRL.Agent.VIAgent import VIAgent, PVIAgent, SPVIAgent, PesBanditAgent

import numpy as np
import math

class PesBandit(offlineTabularBase):

    def set_agent(self):
        self.agent = PesBanditAgent(self.n_states, self.n_actions, self.epLen)

class VI(offlineTabularBase):

    def set_agent(self):
        self.agent = VIAgent(self.n_states, self.n_actions, self.epLen)

class PVI(offlineTabularBase):

    def __init__(self, name, states, actions, epLen, scaling=1., max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf, is_eval = False):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        self.scaling = scaling
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward
        super().__init__(name, states, actions, epLen, is_eval = False)

    def set_agent(self):
        self.agent = PVIAgent(self.n_states, self.n_actions, self.epLen, scaling=self.scaling,
                              max_step_reward=self.max_step_reward, min_step_reward=self.min_step_reward, abs_max_ep_reward=self.abs_max_ep_reward)

class SPVI(offlineTabularBase):

    def __init__(self, name, states, actions, epLen, bpolicy, scaling=1., max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf, is_eval = False):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        self.scaling = scaling
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward
        super().__init__(name, states, actions, epLen, is_eval=False)
        self.bpolicy = bpolicy
        self._extract_bpolicy()

    def _extract_bpolicy(self):
        bpolicy = {}
        for timestep in range(self.agent.epLen):
            for state in range(self.agent.nState):
                probs = (np.zeros(self.agent.nAction, dtype=np.float32))
                for action in range(self.agent.nAction):
                    probs[action] = self.bpolicy._get_action_prob(self.rev_state_dict[state], self.rev_action_dict[action], timestep)
                bpolicy[state, timestep] = probs
        self.agent._update_bpolicy(bpolicy)

    def set_agent(self):
        self.agent = SPVIAgent(self.n_states, self.n_actions, self.epLen, scaling=self.scaling,
                               max_step_reward = self.max_step_reward, min_step_reward = self.min_step_reward, abs_max_ep_reward = self.abs_max_ep_reward)

