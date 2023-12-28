from OfflineSRL.OfflineEvaluators.offlineBaseEvaluator import offlineTabularBaseEvaluator
from OfflineSRL.Agent.EvalAgent import StandardPesEvalAgent, SelectivelyPessimisticEvalAgent, SelectivePessimisticUpdate


import numpy as np
import math

class StandardPesEval(offlineTabularBaseEvaluator):

    # def __init__(self, name, states, actions, epLen, evalpolicy):
    #     super().__init__(name, states, actions, epLen, evalpolicy)

    def set_agent(self):
        self.agent = StandardPesEvalAgent(self.n_states, self.n_actions, self.epLen, state_dict=self.state_dict, rev_state_dict=self.rev_state_dict)

class SelectivePesEval(offlineTabularBaseEvaluator):

    def __init__(self, name, states, actions, epLen, evalpolicy, bpolicy, is_eval = True, is_finite = False):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        
        super().__init__(name, states, actions, epLen, evalpolicy, is_eval=is_eval, is_finite = is_finite)
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
        self.agent = SelectivePessimisticUpdate(self.n_states, self.n_actions, self.epLen, state_dict=self.state_dict, rev_state_dict=self.rev_state_dict)

class SPVIEval(offlineTabularBaseEvaluator):

    def __init__(self, name, states, actions, epLen, evalpolicy, bpolicy, data_splitting, delta, nTrajectories, epsilon1, epsilon2, epsilon3, scaling=1., max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf, is_eval = True, is_finite = False):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        self.scaling = scaling
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward
        self.data_splitting = data_splitting
        self.delta = delta
        self.nTrajectories = nTrajectories
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        super().__init__(name, states, actions, epLen, evalpolicy, is_eval=is_eval, is_finite = is_finite)
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
        self.agent = SelectivelyPessimisticEvalAgent(self.n_states, self.n_actions, self.epLen,nTrajectories = self.nTrajectories,delta= self.delta, data_splitting=self.data_splitting, epsilon1=self.epsilon1, epsilon2= self.epsilon2, epsilon3=self.epsilon3, scaling=self.scaling,state_dict=self.state_dict,rev_state_dict=self.rev_state_dict,
                               max_step_reward = self.max_step_reward, min_step_reward = self.min_step_reward, abs_max_ep_reward = self.abs_max_ep_reward)
