"""
offlineTabularBaseEvaluator builds on offlineTabularBase with a few differences:
- Takes an additional input: 
    - Takes evaluation policy (evalpolicy) as input.
- Has a few additional methods:
    - get_interval:
        - Returns intervals for evaluation policy.
    - _extract_evalpolicy:
        - extracts the evaluation policy and passes it to the agent.
    - update_agent_intervals:
        - Updates self.interval.
        - It is now run when the fit method is called.
    - New fit method:
        - input: MDPDataset
        - Fit process:
            - Maps the MDPDataset to format acceptable by the agent (using set_map).
            - Updates agent with this mapped data.
            - Then updates the agent's policy.

offlineTabularBase has the following methods:
- set_map:
    - Defines a mapping from original states to state indices.
    - Defines a mapping from original actions to action indices.
    - Also defines the reverse mappings.
- set_agent:
    - Set a tabular RL agent.
- update_agent_obs:
    - inputs: obs, action, reward, newObs, pContinue, h
    - Add this datapoint to the RL agent.
- update_agent_policy:
    - Use agent.update_policy() to update the policy.
- fit:
    - input: MDPDataset
    - Fit process:
        - Maps the MDPDataset to format acceptable by the agent (using set_map).
        - Updates agent with this mapped data.
        - Then updates the agent's policy.
- act:
    - Input: state, timestep.
    - Return action by an action from the agent (self.agent.pick_action) -- Then apply rev_action_dict to get the original action.
- policy:
    - Input: state, timestep.
    - Returns output of self.act(state, timestep).
- Methods related to name: set_name, get_name, __str__.
"""

from OfflineSRL.OfflineLearners.offlineBase import offlineTabularBase
import math

class offlineTabularBaseEvaluator(offlineTabularBase):

    def __init__(self, name, states, actions, epLen, evalpolicy):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        self.evalpolicy = evalpolicy
        super().__init__(name, states, actions, epLen)
        self._extract_evalpolicy()

    def _extract_evalpolicy(self):
        evalpolicy = {}
        for timestep in range(self.agent.epLen):
            for state in range(self.agent.nState):
                probs = (np.zeros(self.agent.nAction, dtype=np.float32))
                for action in range(self.agent.nAction):
                    probs[action] = self.evalpolicy._get_action_prob(self.rev_state_dict[state], self.rev_action_dict[action], timestep)
                evalpolicy[state, timestep] = probs
        self.agent._update_evalpolicy(evalpolicy)
    
    def update_agent_intervals(self):
        self.agent.update_intervals()
    
    def get_interval(self):
        return self.agent.interval
    
    def fit(self, mdpdataset):
        self.n_episodes = len(mdpdataset.episodes)
        for episode in mdpdataset.episodes:
            h = 1
            for transition in episode.transitions:
                obs = self.state_dict[str(transition.observation)]
                action = self.action_dict[str(transition.action)]
                reward = transition.reward
                newObs = self.state_dict[str(transition.next_observation)]
                self.update_agent_obs(obs, action, reward, newObs, pContinue = 1, h = h)
                h = h+1
            self.update_agent_obs(newObs, action, reward, newObs, pContinue=0, h = h)
        self.update_agent_intervals()
