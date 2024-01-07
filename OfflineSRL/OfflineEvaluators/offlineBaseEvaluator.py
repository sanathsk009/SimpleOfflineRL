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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error 
class offlineTabularBaseEvaluator(offlineTabularBase):

    def __init__(self, name, states, actions, epLen, evalpolicy, is_eval = True, is_finite = False):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        self.evalpolicy = evalpolicy
        super().__init__(name, states, actions, epLen, is_eval=is_eval, is_finite = is_finite)
        self._extract_evalpolicy()

    def _extract_evalpolicy(self):
        # evalpolicy = {}
        # for timestep in range(self.agent.epLen):
        #     for state in range(self.agent.nState):
        #         probs = (np.zeros(self.agent.nAction, dtype=np.float32))
        #         for action in range(self.agent.nAction):
        #             probs[action] = self.evalpolicy._get_action_prob(self.rev_state_dict[state], self.rev_action_dict[action], timestep)
        #         evalpolicy[state, timestep] = probs
        self.agent._update_evalpolicy(self.evalpolicy)
    
    def update_agent_intervals(self, trajectory_states, reward_predictor, transition_predictor):
        if self.name == "pvi":
            self.agent.update_intervals(reward_predictor, transition_predictor, self.is_finite)
        else:
            self.agent.update_intervals(trajectory_states, reward_predictor, transition_predictor, self.is_finite)

    def predict_probability(self):
        temp = {}
        # self.P_prior = {}

        for state in range(self.n_states):
            for action in range(self.n_actions):
                
                temp[state, action] = (
                    np.ones(self.n_states))

        for i in self.agent.dataset:
            temp[self.state_dict[str(i[0])], i[1]][self.state_dict[str(i[3])]]+=1
        # for i in self.dataset:
        #     self.P_prior[self.rev_state_dict(i[0]), i[1]][self.rev_state_dict(i[3])] =  
        # for s in range(self.nState):
        #     for a in range(self.nAction):
        #         P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        for i in range(len(self.agent.dataset)):
            self.agent.dataset[i].append(temp[self.state_dict[str(self.agent.dataset[i][0])], self.agent.dataset[i][1]][self.state_dict[str(self.agent.dataset[i][3])]]/ np.sum(temp[self.state_dict[str(self.agent.dataset[i][0])], self.agent.dataset[i][1]]))

        df = pd.DataFrame(self.agent.dataset, columns =['State', 'Action', 'Reward', 'Next State', 'Transition Probability'])
        df_binary = df[['State', 'Action', 'Reward']]
        print(df)
        print("this is input df with transition prob")
        df_binary.columns = ['State', 'Action', 'Reward']
        df3 = pd.DataFrame(df_binary['State'].to_list(), columns=['first','second'])
        df4 = pd.DataFrame(df['Next State'].to_list(), columns=['new first', 'new second'])
        X = pd.concat([df3, df_binary['Action'], df4], axis=1)
        y = df['Transition Probability']

        print(X)
        print("this is the transition predictor input")
        # tp = pd.concat([X, y], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
        regr = MLPRegressor(hidden_layer_sizes=(100, 10), random_state=1, max_iter=1000).fit(X_train, y_train)
        # 
        print(mean_squared_error(regr.predict(X_test), y_test))

        print("transition mean squared error")
        return regr
    
    def get_interval(self):
        return self.agent.interval
    
    def fit(self, mdpdataset):
        self.n_episodes = len(mdpdataset.episodes)
        trajectory_states = [] #for getting first state of every trajectory
        self.agent.dataset = []
        for episode in mdpdataset.episodes:
            h = 1
            temp_trajectory_state = []
            for transition in episode.transitions:
                obs = self.state_dict[str(transition.observation)]
                print("old state" + str(obs))
                temp_trajectory_state.append(obs)
                print("this is the main action coming in")
                print(str(transition.action))
                action = self.action_dict[str(transition.action)] 
                print("action" + str(action))
                reward = transition.reward
                newObs = self.state_dict[str(transition.next_observation)]
                print("new state" + str(newObs))
                self.agent.dataset.append([transition.observation, action, reward, transition.next_observation])
                self.update_agent_obs(obs, action, reward, newObs, pContinue = 1, h = h)
                h = h+1
            h-=1
            # self.update_agent_obs(newObs, action, reward, newObs, pContinue=0, h = h)
            trajectory_states.append(temp_trajectory_state)
        reward_predictor = None
        transition_predictor = None
        if not self.is_finite:
            print("here")
            reward_predictor = self.agent.predict_reward()
            transition_predictor = self.predict_probability()
        self.update_agent_intervals(trajectory_states, reward_predictor, transition_predictor)
