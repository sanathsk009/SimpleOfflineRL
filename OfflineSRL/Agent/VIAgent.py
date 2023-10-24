"""
Agents implement the core policy learning algorithm. These agents are used by offline learners to describe the complete offline learner module.

Class objects:
- PesBanditAgent
    - Implements an agent with standard bandit pessimism.
    - Inherits FiniteHorizonTabularAgent.
    - gen_bandit_bonus:
        Generates statistical bonusus without dependence on horizon length.
    - computeRVals:
        Creates pessimistic Q-value equivalents only from Rewards and pessimism.

- VIAgent
    - Implements an agent with standard value iteration.
    - Inherits FiniteHorizonTabularAgent.
    - update_policy:
        - No used input.
        - Updates self.qVals and self.qMax using self.compute_qVals(R_hat, P_hat).

- PVIAgent
    - Implements an agent with pessimistic value iteration.
    - Inherits FiniteHorizonTabularAgent.
    - Has an additional parameter (self.scaling) to control the size of bonus.
    - get_bonus:
        - No used input.
        - Recovers an (s,a) count: R_sum = self.R_prior[s,a][1].
        - Computes a bonus proportional to: self.scaling * np.sqrt(np.log(self.nState * self.nAction * self.epLen)/R_sum).
    - update_policy:
        - No used input.
        - Updates self.qVals and self.qMax using self.compute_qVals_pes(R_hat, P_hat, pessimism).
        - The method compute_qVals_pes is implemented in BaseFiniteHorizonTabularAgent.

- SPVIAgent
    - Implements an agent with selectively pessimistic value iteration.
    - Inherits FiniteHorizonTabularAgent.
    - Has an additional parameter (self.scaling) to control the size of bonus.
    - Also has a parameter (self.maxeprew) to control size of maximum reward at any step. Intent was to further bound pessimism.
        - Didn't seem to make a difference.
        - Currently not being used.
    - get_bonus:
        - No used input.
        - Recovers an (s,a) count: R_sum = self.R_prior[s,a][1].
        - Computes a bonus proportional to: self.scaling * np.sqrt(np.log(self.nState * self.nAction * self.epLen)/R_sum).
    - _update_bpolicy:
        - A method for SPVI (the offlineBase learner for this agent) to input the behavioral policy.
        - The SPVI method can extract the behavioral policy and encode it into the tabular format.
        - It then uses this method to feed the extracted policy to the agent.
    - _get_shift:
        - No input.
        - Extracts shifts from estimated tabular models.
    - update_obs:
        - Still unchanged from the base method implemented in BaseFiniteHorizonTabularAgent.
        - Goal is to add a modification where we directly learn shift in an online fashion.
    - update_policy:
        - No used input.
        - Updates self.qVals and self.qMax using self.compute_qVals_spes(R_hat, P_hat, pessimism, shift).
    - compute_qVals_spes:
        - Takes as input: R, P, pessimism, shift.
        - Computes optimistic, pessimistic, and standard QVals: Their differences help estimate uncertainty.
        - GVals are our selectively pessimistic value estimates.
"""

import numpy as np
import math

from OfflineSRL.Agent.BaseFiniteHorizonTabularAgent import FiniteHorizonTabularAgent

class PesBanditAgent(FiniteHorizonTabularAgent):

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super().__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.scaling = scaling

    def gen_bandit_bonus(self, h=1):
        ''' Generate the sqrt(n) bonus for UCBVI '''
        R_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                #R_bonus[s, a] = self.scaling * np.sqrt(2. * np.log(2 + h) / R_sum)
                R_bonus[s, a] = self.scaling * np.sqrt(np.log(self.nState * self.nAction) / R_sum)

        return R_bonus

    def compute_RVals(self, R, P, pessimism):
        '''
        Computes Q values solely based on reward R.

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = max(R[s, a] - pessimism[s, a], 0)

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def update_policy(self, h=False):
        '''
        Update Q values via Gaussian PSRL.
        This performs value iteration but with additive Gaussian noise.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Pessimistic bonus.
        pessimism = self.gen_bandit_bonus(h)

        # Form approximate Q-value estimates
        qVals, qMax = self.compute_RVals(R_hat, P_hat, pessimism)

        self.qVals = qVals
        self.qMax = qMax


class VIAgent(FiniteHorizonTabularAgent):

    def update_policy(self, h=False):
        '''
        Update Q values via Gaussian PSRL.
        This performs value iteration but with additive Gaussian noise.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Form approximate Q-value estimates
        qVals, qMax = self.compute_qVals(R_hat, P_hat)

        self.qVals = qVals
        self.qMax = qMax

class PVIAgent(FiniteHorizonTabularAgent):

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1., max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super().__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.scaling = scaling
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward

    def gen_bonus(self, h=1):
        ''' Generate the sqrt(n) bonus for UCBVI '''
        R_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                #R_bonus[s, a] = self.scaling * np.sqrt(2. * np.log(2 + h) / R_sum)
                R_bonus[s, a] = self.scaling * np.sqrt(np.log(self.nState * self.nAction * self.epLen) / R_sum)

        return R_bonus

    def update_policy(self, h=False):
        '''
        Update Q values via Gaussian PSRL.
        This performs value iteration but with additive Gaussian noise.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Pessimistic bonus.
        pessimism = self.gen_bonus(h)

        # Form approximate Q-value estimates
        qVals, qMax = self.compute_qVals_pes(R_hat, P_hat, pessimism)

        self.qVals = qVals
        self.qMax = qMax

    def compute_qVals_pes(self, R, P, pessimism):
        '''
        Compute the Q values for a given R, P estimates + bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            pessimism - pessimism[s,a] = bonus for state action visitation.

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
                    min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value
                    qVals[s, j][a] = R[s, a] - pessimism[s, a] + np.dot(P[s, a], qMax[j + 1])
                    qVals[s, j][a] = max(qVals[s, j][a], min_val) # Can't be lower than min_val.
                    qVals[s, j][a] = min(qVals[s, j][a], max_val) # Can't be larger than max_val.
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

class SPVIAgent(FiniteHorizonTabularAgent):

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.,  max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super().__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.scaling = scaling
        self.scaling = scaling
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward
        """
        self.shift_count = {}
        self.shift_prior = {}
        for timestep in range(epLen):
            for state in range(nState):
                for action in range(nAction):
                    self.shift_prior[state, action, timestep] = (np.zeros(self.nState, dtype=np.float32))
                    self.shift_count[state,action,timestep] = (np.zeros(self.nState, dtype=np.float32))
        """

    def _update_bpolicy(self, policy):
        self.bpolicy = policy

    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''
        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)
        
        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1
            """
            self.shift_count[oldState, action, h] += 1
            self.shift_prior[oldState, action, h][newState] += 1
            for action_prime in in range(self.nAction):
                self.shift_prior[oldState, action_prime, h][newState] += -self.bpolicy[oldState,h][action_prime]
                self.shift_count[oldState, action_prime, h] 
            """

    def gen_bonus(self, h=1):
        ''' Generate the sqrt(n) bonus for UCBVI '''
        R_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = self.scaling * np.sqrt(np.log(self.nState * self.nAction * self.epLen) / R_sum)

        return R_bonus

    def _get_shift(self):
        R_hat, P_hat = self.map_mdp()
        P_b = {}
        shift = {}

        for timestep in range(self.epLen):
            for state in range(self.nState):
                P_b[state, timestep] = np.zeros(self.nState, dtype=np.float32)
                for action in range(self.nAction):
                    P_b[state, timestep] += self.bpolicy[state, timestep][action] * P_hat[state, action]
                # We now have behavioral transition
                for action in range(self.nAction):
                    shift[state, action, timestep] = P_hat[state, action] - P_b[state, timestep]

        self.shift_prior = shift
        return shift


    def update_policy(self, h=False):
        '''
        Update Q values via Gaussian PSRL.
        This performs value iteration but with additive Gaussian noise.
        '''
        shift = self._get_shift()
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Purely Gaussian perturbations
        pessimism = self.gen_bonus(h)

        # Form approximate Q-value estimates
        GVals = self.compute_qVals_spes(R_hat, P_hat, pessimism, shift)

        self.qVals = GVals

    def compute_qVals_spes(self, R, P, pessimism, shift):
        '''
        Compute the Q values for a given R, P estimates + bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            pessimism - pessimism[s,a] = bonus for state action visitation.

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qPol = {}

        PesqVals = {}
        PesqPol = {}

        OptqVals = {}
        OptqPol = {}

        GVals = {}

        qPol[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        PesqPol[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        OptqPol[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qPol[j] = np.zeros(self.nState, dtype=np.float32)
            PesqPol[j] = np.zeros(self.nState, dtype=np.float32)
            OptqPol[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
                PesqVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
                OptqVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
                GVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
                    min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value

                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qPol[j + 1])
                    PesqVals[s, j][a] = R[s, a] - pessimism[s, a] + np.dot(P[s, a], PesqPol[j + 1])
                    PesqVals[s, j][a] = max(PesqVals[s, j][a], min_val) # Can't be lower than min_val
                    PesqVals[s, j][a] = min(PesqVals[s, j][a], max_val) # Can't be larger than max_val

                    OptqVals[s, j][a] = R[s, a] + pessimism[s, a] + np.dot(P[s, a], OptqPol[j + 1])
                    OptqVals[s, j][a] = max(OptqVals[s, j][a], min_val)
                    OptqVals[s, j][a] = min(OptqVals[s, j][a], max_val)

                    #GVals[s, j][a] = R[s, a] - pessimism[s, a] + np.dot(P[s, a], qPol[j + 1]) \
                    #                    - min(np.dot(np.maximum(shift[s, a, j],0), OptqPol[j+1] - qPol[j+1]) + np.dot(np.maximum(-shift[s, a, j],0), qPol[j+1] - PesqPol[j+1]), self.maxeprew)

                    """
                    Line 1 in GVals: Standard value iteration with reward pessimism.
                    Line 2 in GVals: Propagating underestimation error. (a negative term)
                    Line 3 in GVals: Propagating overestimation error. (also a negative term)
                    """
                    """
                    # Without bounds based on cumulative reward.
                    GVals[s, j][a] = R[s, a] - pessimism[s, a] + np.dot(P[s, a], qPol[j + 1]) \
                                     - np.dot(np.maximum(shift[s, a, j], 0), OptqPol[j + 1] - qPol[j + 1]) \
                                     - np.dot(np.maximum(-shift[s, a, j], 0), qPol[j + 1] - PesqPol[j + 1])
                    """

                    max_abs_val = max(max_val, -min_val)
                    GVals[s, j][a] = R[s, a] - pessimism[s, a] + np.dot(P[s, a], qPol[j + 1]) \
                                     - min(np.dot(np.maximum(shift[s, a, j], 0), OptqPol[j + 1] - qPol[j + 1]) \
                                     + np.dot(np.maximum(-shift[s, a, j], 0), qPol[j + 1] - PesqPol[j + 1]), max_abs_val)

                optarm = np.argmax(GVals[s,j])
                qPol[j][s] = qVals[s,j][optarm]
                PesqPol[j][s] = PesqVals[s, j][optarm]
                OptqPol[j][s] = OptqVals[s, j][optarm]
                #qMax[j][s] = np.max(qVals[s, j])

        return GVals


