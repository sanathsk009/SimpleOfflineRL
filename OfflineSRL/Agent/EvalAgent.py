import numpy as np
import math

from OfflineSRL.Agent.BaseFiniteHorizonTabularAgent import FiniteHorizonTabularAgent

class StandardPesEvalAgent(FiniteHorizonTabularAgent):

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

    def compute_extremeVals_evalpolicy(self, R, P, pessimism):
        '''
        Compute extreme values for evaluation policy for a given R, P estimates + bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            pessimism - pessimism[s,a] = bonus for state action visitation.

        Returns:
            OptVals - OptVals[state, timestep] is the optimistic value for evalution policy at state-time pair.
            PesVals - PesVals[state, timestep] is the pessimistic value for evalution policy at state-time pair.
        '''
        OptVals = {}
        PesVals = {}

        OptVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        PesVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            OptVals[j] = np.zeros(self.nState, dtype=np.float32)
            PesVals[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                OptVals[j][s] = 0
                PesVals[j][s] = 0

                for a in range(self.nAction):
                    max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
                    min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value
                    
                    pes_action_contribution = R[s, a] - pessimism[s, a] + np.dot(P[s, a], PesVals[j + 1])
                    pes_action_contribution = max(pes_action_contribution, min_val) # Can't be lower than min_val.
                    pes_action_contribution = min(pes_action_contribution, max_val) # Can't be larger than max_val.
                    PesVals[j][s] += self.evalpolicy[s, j][a] * pes_action_contribution

                    opt_action_contribution = R[s, a] + pessimism[s, a] + np.dot(P[s, a], OptVals[j + 1])
                    opt_action_contribution = max(opt_action_contribution, min_val) # Can't be lower than min_val.
                    opt_action_contribution = min(opt_action_contribution, max_val) # Can't be larger than max_val.
                    OptVals[j][s] += self.evalpolicy[s, j][a] * opt_action_contribution

        return OptVals, PesVals
    
    def update_intervals():
        '''
        Update intervals via UCBVI.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Pessimistic bonus.
        pessimism = self.gen_bonus(h)

        # Form approximate Q-value estimates
        OptVals, PesVals = self.compute_extremeVals_evalpolicy(R_hat, P_hat, pessimism)

        self.OptVals = OptVals
        self.PesVals = PesVals

        emp_dist = self.behavioral_state_distribution[0]
        lower_bound = np.dot(emp_dist, PesVals[0])/np.sum(emp_dist)
        upper_bound = np.dot(emp_dist, OptVals[0])/np.sum(emp_dist)
        self.interval = (lower_bound, upper_bound)

class SelectivelyPessimisticEvalAgent(FiniteHorizonTabularAgent):

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

    def compute_rectifiedVals_evalpolicy(self, R, P, pessimism, shift):
        '''
        Still editing.
        Compute the Q values for a given R, P estimates + bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            pessimism - pessimism[s,a] = bonus for state action visitation.

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        OptVals = {}
        PesVals = {}

        OptVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        PesVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        RVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        avg_uncertainty = 0
        shift_uncertainty = 0 # Need to add uncertainties

        for i in range(self.epLen):
            j = self.epLen - i - 1
            OptVals[j] = np.zeros(self.nState, dtype=np.float32)
            PesVals[j] = np.zeros(self.nState, dtype=np.float32)
            RVals[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                OptVals[j][s] = 0
                PesVals[j][s] = 0

                for a in range(self.nAction):
                    max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
                    min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value
                    
                    pes_action_contribution = R[s, a] - pessimism[s, a] + np.dot(P[s, a], PesVals[j + 1])
                    pes_action_contribution = max(pes_action_contribution, min_val) # Can't be lower than min_val.
                    pes_action_contribution = min(pes_action_contribution, max_val) # Can't be larger than max_val.
                    PesVals[j][s] += self.evalpolicy[s, j][a] * pes_action_contribution

                    opt_action_contribution = R[s, a] + pessimism[s, a] + np.dot(P[s, a], OptVals[j + 1])
                    opt_action_contribution = max(opt_action_contribution, min_val) # Can't be lower than min_val.
                    opt_action_contribution = min(opt_action_contribution, max_val) # Can't be larger than max_val.
                    OptVals[j][s] += self.evalpolicy[s, j][a] * opt_action_contribution

                    rectified_action_contribution = R[s, a] + np.dot(P[s, a], RVals[j + 1])
                    rectified_action_contribution = max(rectified_action_contribution, pes_action_contribution)
                    rectified_action_contribution = min(rectified_action_contribution, opt_action_contribution)
                    RVals[j][s] += self.evalpolicy[s, j][a] * rectified_action_contribution

                    avg_uncertainty += self.behavioral_state_distribution[j][s] * self.evalpolicy[s, j][a] * pessimism[s, a]

        return OptVals, PesVals, RVals, avg_uncertainty
        
    
    def update_intervals():
        '''
        Update intervals via UCBVI.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Pessimistic bonus.
        pessimism = self.gen_bonus(h)

        # Get shift
        shift = self._get_shift()

        # Form approximate Q-value estimates
        OptVals, PesVals, RVals, avg_uncertainty = self.compute_extremeVals_evalpolicy(R_hat, P_hat, pessimism, shift)

        self.OptVals = OptVals
        self.PesVals = PesVals

        emp_dist = self.behavioral_state_distribution[0]
        lower_bound = np.dot(emp_dist, PesVals[0])/np.sum(emp_dist)
        upper_bound = np.dot(emp_dist, OptVals[0])/np.sum(emp_dist)
        self.interval = (lower_bound, upper_bound)