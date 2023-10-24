"""
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

from OfflineSRL.Agent.BaseFiniteHorizonTabularAgent import FiniteHorizonTabularAgent
import numpy as np

class offlineTabularBase(object):

    def __init__(self, name, states, actions, epLen, is_eval = False):
        self.name = name
        if is_eval:
            self.states = states
        else:
            self.states = list(states)
        self.actions = list(actions) # Just in case we're given a numpy array (like from Atari).
        self.epLen = epLen
        self.is_eval = is_eval
        self.set_map()
        self.set_agent()

    def set_map(self):
        # Mapping from original states to index.
        self.state_dict = {}
        # Mapping from state index to original states.
        self.rev_state_dict = {}
        # Mapping from original action to index.
        self.action_dict = {}
        # Mapping from action index to original actions.
        self.rev_action_dict = {}
        self.n_states = 0
        self.n_actions = 0


        if self.is_eval:
            self.state_dict[str(np.array([-1, -1]))] = 0
            self.rev_state_dict[0] = np.array([-1, -1])
            for state in range(1, self.states+1):
                self.state_dict[str(np.array([state, 0]))] = (2*state)-1
                self.state_dict[str(np.array([state, 1]))] = (2*state)
                self.rev_state_dict[(2*state)-1] = np.array([state, 0])
                self.rev_state_dict[2*state] = np.array([state, 1])

        else:
            for state in self.states:
                if str(state) not in self.state_dict:
                        self.state_dict[str(state)] = self.n_states
                        self.rev_state_dict[self.n_states] = state
                        self.n_states += 1
        if self.is_eval:
            print(self.state_dict)
        for action in self.actions:
            if str(action) not in self.action_dict:
                self.action_dict[str(action)] = self.n_actions
                self.rev_action_dict[self.n_actions] = action
                self.n_actions += 1

        if self.is_eval:
            self.n_states = (2*self.states) + 1

    def set_agent(self):
        self.agent = FiniteHorizonTabularAgent(self.n_states, self.n_actions, self.epLen)

    def update_agent_obs(self, obs, action, reward, newObs, pContinue, h):
        self.agent.update_obs(obs, action, reward, newObs, pContinue, h)

    def update_agent_policy(self):
        self.agent.update_policy()

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        return {}

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
                h+=1
            h-=1
            self.update_agent_obs(newObs, action, reward, newObs, pContinue=0, h = h)
           
        self.update_agent_policy()

    def act(self, state, timestep):
        '''
        Args:
            state (State): see StateClass.py
            reward (float): the reward associaated with arriving in state @state.

        Returns:
            (str): action.
        '''

        state_index = self.state_dict[str(state)]
        return self.rev_action_dict[self.agent.pick_action(state_index, timestep)]

    def policy(self, state, timestep = 0):
        return self.act(state, timestep)

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __str__(self):
        return str(self.name)
