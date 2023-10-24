import numpy as np
import math

from OfflineSRL.Agent.BaseFiniteHorizonTabularAgent import FiniteHorizonTabularAgent

class StandardPesEvalAgent(FiniteHorizonTabularAgent):

    def __init__(self, nState, nAction, epLen, state_dict, rev_state_dict, 
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1., max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super().__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.scaling = scaling
        self.state_dict = state_dict
        self.rev_state_dict = rev_state_dict
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward

    def gen_bonus(self):
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
        test = [tuple(i[1]) for i in list(self.rev_state_dict.items())]
        test = np.array(test)
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
                    final_input = np.concatenate((np.broadcast_to(self.rev_state_dict[s], (self.nState, 2)), np.broadcast_to(np.int64(a),(self.nState, 1)), test), axis = 1)
                    pes_action_contribution = R.predict(np.array([self.rev_state_dict[s][0], self.rev_state_dict[s][1], a]).reshape(1, -1))[0] - pessimism[s, a] + np.dot(P.predict(final_input), PesVals[j + 1])
                    print(pessimism[s,a])
                    print("this is pessimism")
                    pes_action_contribution = max(pes_action_contribution, min_val) # Can't be lower than min_val.
                    pes_action_contribution = min(pes_action_contribution, max_val) # Can't be larger than max_val.
                    PesVals[j][s] += self.evalpolicy[s, j][a] * pes_action_contribution

                    opt_action_contribution = R.predict(np.array([self.rev_state_dict[s][0], self.rev_state_dict[s][1], a]).reshape(1, -1))[0] + pessimism[s, a] + np.dot(P.predict(final_input), OptVals[j + 1])
                    opt_action_contribution = max(opt_action_contribution, min_val) # Can't be lower than min_val.
                    opt_action_contribution = min(opt_action_contribution, max_val) # Can't be larger than max_val.
                    OptVals[j][s] += self.evalpolicy[s, j][a] * opt_action_contribution

        return OptVals, PesVals
    
    def update_intervals(self, R_hat, P_hat):
        '''
        Update intervals via UCBVI.
        '''
        # Output the MAP estimate MDP
        # R_hat, P_hat = self.map_mdp()

        # Pessimistic bonus.
        pessimism = self.gen_bonus()

        # Form approximate Q-value estimates
        OptVals, PesVals = self.compute_extremeVals_evalpolicy(R_hat, P_hat, pessimism)

        self.OptVals = OptVals
        self.PesVals = PesVals

        emp_dist = self.behavioral_state_distribution[0]
        lower_bound = np.dot(emp_dist, PesVals[0])/np.sum(emp_dist)
        upper_bound = np.dot(emp_dist, OptVals[0])/np.sum(emp_dist)
        self.interval = (lower_bound, upper_bound)

    def get_interval(self):
        return self.interval[1]

    def get_total_uncertainty(self):
        return self.total_uncertainty

class SelectivelyPessimisticEvalAgent(FiniteHorizonTabularAgent):

    def __init__(self, nState, nAction, epLen,nTrajectories, delta, data_splitting, epsilon1, epsilon2, epsilon3, state_dict, rev_state_dict, 
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.,  max_step_reward = 1, min_step_reward = -1, abs_max_ep_reward = math.inf):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super().__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.state_dict = state_dict
        self.rev_state_dict = rev_state_dict
        self.scaling = scaling
        self.scaling = scaling
        self.max_step_reward = max_step_reward
        self.min_step_reward = min_step_reward
        self.abs_max_ep_reward = abs_max_ep_reward
        self.nTrajectories = nTrajectories
        self.delta = delta
        self.data_splitting = data_splitting
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
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

    def gen_bonus(self, R):
        ''' Generate the sqrt(n) bonus for UCBVI '''
        R_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = self.scaling * np.sqrt(np.log(self.nState * self.nAction * self.epLen) / R_sum)

        return R_bonus

    def _get_shift(self, R, P):
        # R_hat, P_hat = self.map_mdp()
        P_b = {}
        shift = {}
        test = [tuple(i[1]) for i in list(self.rev_state_dict.items())]
        test = np.array(test)
        
        for timestep in range(self.epLen):
            for state in range(self.nState):
                P_b[state, timestep] = np.zeros(self.nState, dtype=np.float32)
                for action in range(self.nAction):
                    final_input = np.concatenate((np.broadcast_to(self.rev_state_dict[state], (self.nState, 2)), np.broadcast_to(np.int64(action),(self.nState, 1)), test), axis = 1)
                    P_b[state, timestep] += self.bpolicy[state, timestep][action] * P.predict(final_input)
                # We now have behavioral transition
                for action in range(self.nAction):
                    final_input = np.concatenate((np.broadcast_to(self.rev_state_dict[state], (self.nState, 2)), np.broadcast_to(np.int64(action),(self.nState, 1)), test), axis = 1)
                
                    shift[state, action, timestep] =  P.predict(final_input) - P_b[state, timestep]
                    print('hereferfe')
                    print(shift[state, action, timestep])

        self.shift_prior = shift
        return shift

    def compute_rectifiedVals_evalpolicy(self, R, P, pessimism, shift, OptVals_uncertainty, PesVals_uncertainty, trajectory_states):
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
        V_average = {}
        test = [tuple(i[1]) for i in list(self.rev_state_dict.items())]
        test = np.array(test)
        OptVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        PesVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        V_average[self.epLen] = np.zeros(self.nState, dtype = np.float32)
        # avg_uncertainty = 0
        # shift_uncertainty = 0 # Need to add uncertainties
        total_uncertainty = 0
        uncertainty_opt_pes = {}
        uncertainty_opt_pes[self.epLen] = np.zeros(self.nState, dtype = np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            uncertainty_opt_pes[j] = np.zeros(self.nState, dtype = np.float32)
            for s in range(self.nState):
                uncertainty_opt_pes[j][s] = (OptVals_uncertainty[j][s] - PesVals_uncertainty[j][s])
        
        for i in range(self.epLen):
            j = self.epLen - i - 1
            OptVals[j] = np.zeros(self.nState, dtype=np.float32)
            PesVals[j] = np.zeros(self.nState, dtype=np.float32)
            # RVals[j] = np.zeros(self.nState, dtype=np.float32)
            V_average[j] = np.zeros(self.nState, dtype = np.float32)

            for s in range(self.nState):
                OptVals[j][s] = 0
                PesVals[j][s] = 0
                temp_sum = 0
                for a in range(self.nAction):

                    final_input = np.concatenate((np.broadcast_to(self.rev_state_dict[s], (self.nState, 2)), np.broadcast_to(np.int64(a),(self.nState, 1)), test), axis = 1)
                    temp_sum+=self.evalpolicy[s, j][a]*(R.predict(np.array([self.rev_state_dict[s][0], self.rev_state_dict[s][1], a]).reshape(1, -1))[0] + np.dot(P.predict(final_input), V_average[j+1]))
                V_average[j][s] = max(PesVals_uncertainty[j][s], min(OptVals_uncertainty[j][s], temp_sum))
            
            max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
            min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value
            first_term_average = 0
            first_term_max = 0

            for trajectory in range(self.nTrajectories):
                expected_first_term = 0
                for action in range(self.nAction):
                   
                    # print(str(action) + "Action----------")
                    # print(str(j)+"horizonn-----------")
                    # print(str(trajectory) + "trajectory---")
                    # print(str(trajectory_states[trajectory][j]) + "trajectory state")
                    expected_first_term+=(((self.evalpolicy[trajectory_states[trajectory][j],j][action])**2)/(self.bpolicy[trajectory_states[trajectory][j],j][action]))

                first_term_average+=expected_first_term
                if expected_first_term > first_term_max:
                    first_term_max = expected_first_term
            first_term_average/=self.nTrajectories
            # print("log term " + str(math.sqrt(math.log((10*self.epLen)/self.delta)/(2*self.nTrajectories))))
            # print("maxc term " + str(first_term_max))
            second_term = first_term_max*math.sqrt(math.log((10*self.epLen)/self.delta)/(2*self.nTrajectories))
            first_term_final = math.sqrt(first_term_average + second_term)*(math.sqrt(self.epsilon1) + math.sqrt(self.epsilon2))
            v_max = 0
            for temp in range(self.nState):
                if OptVals_uncertainty[j+1][temp] > v_max:
                    v_max = OptVals_uncertainty[j+1][temp]
            second_term_final = v_max*math.sqrt(self.epsilon3)

            # for state in range(self.nState):
            #     for action in range(self.nAction):
                    # shift[state, action, j]*=self.evalpolicy[state, j][action]
            third_term_final = 0
            for trajectory in range(self.nTrajectories):
                temp_sum = 0
                for action in range(self.nAction):
                    print(uncertainty_opt_pes[j+1])
                    print("importanmt thing to look at")
                    temp_sum+=self.evalpolicy[trajectory_states[trajectory][j], j][action]*np.dot(np.absolute(shift[trajectory_states[trajectory][j], action, j]), uncertainty_opt_pes[j+1])
                third_term_final+=temp_sum
            third_term_final/=self.nTrajectories
            fourth_term_final = 2*v_max*math.sqrt(math.log((10*self.epLen)/self.delta)/(2*self.nTrajectories))
            uncertainty = first_term_final + second_term_final + third_term_final + fourth_term_final
            # print("HORIZON ---------" + str(j))
            print("first term ---- " + str(first_term_final))
            print("second term ---- " + str(second_term_final))
            print("third term ---- " + str(third_term_final))
            print("fourth term ---- " + str(fourth_term_final))
            total_uncertainty += uncertainty
        total_uncertainty+=v_max*math.sqrt(math.log((10)/self.delta)/(2*self.nTrajectories))
        final_average_v_value = 0
        for trajectory in range(0,self.nTrajectories):
            final_average_v_value+=V_average[0][trajectory_states[trajectory][0]]
        final_average_v_value/=self.nTrajectories
                    # pes_action_contribution = R[s, a] - pessimism[s, a] + np.dot(P[s, a], PesVals[j + 1])
                    # pes_action_contribution = max(pes_action_contribution, min_val) # Can't be lower than min_val.
                    # pes_action_contribution = min(pes_action_contribution, max_val) # Can't be larger than max_val.
                    # PesVals[j][s] += self.evalpolicy[s, j][a] * pes_action_contribution

                    # opt_action_contribution = R[s, a] + pessimism[s, a] + np.dot(P[s, a], OptVals[j + 1])
                    # opt_action_contribution = max(opt_action_contribution, min_val) # Can't be lower than min_val.
                    # opt_action_contribution = min(opt_action_contribution, max_val) # Can't be larger than max_val.
                    # OptVals[j][s] += self.evalpolicy[s, j][a] * opt_action_contribution

                    # rectified_action_contribution = R[s, a] + np.dot(P[s, a], RVals[j + 1])
                    # rectified_action_contribution = max(rectified_action_contribution, pes_action_contribution)
                    # rectified_action_contribution = min(rectified_action_contribution, opt_action_contribution)
                    # RVals[j][s] += self.evalpolicy[s, j][a] * rectified_action_contribution

                    # avg_uncertainty += self.behavioral_state_distribution[j][s] * self.evalpolicy[s, j][a] * pessimism[s, a]

        return (final_average_v_value, total_uncertainty)
        
    def uncertainty_prop(self, R, P, pessimism):

        # PesVals = {}
        # OptVals = {}
        # P_eval = {}

        # for temp_state in range(self.nState):
        #     PesVals[temp_state, self.epLen] = 0
        #     OptVals[temp_state, self.epLen] = 0

        # for i in range(self.epLen):
        #     j = self.epLen - i - 1

        #     for s in range(self.nState):
        #         PesVals[s, j] = 0
        #         OptVals[s, j] = 0
        #         temp_expected_reward = 0
        #         temp_pessimism = 0
        #         P_eval[s, j] = np.zeros(self.nState, dtype=np.float32)
        #         for a in range(self.nAction):
                    
        #             temp_expected_reward+=R[s,a]*self.evalpolicy[s, j][a]
        #             temp_pessimism+=pessimism[s,a]*self.evalpolicy[s,j][a]
        #             P_eval[s, j]+= self.evalpolicy[s, j][a] * P[s, a]

        #         temp_dot = 0
        #         for temp_state in range(self.nState):
        #             temp_dot+=P_eval[s, j][temp_state]*PesVals[temp_state, j+1]
                
        #         max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
        #         min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value

        #         PesVals[s,j] = temp_expected_reward - temp_pessimism + temp_dot
        #         PesVals[s, j] = max(PesVals[s, j], min_val) # Can't be lower than min_val
        #         PesVals[s, j] = min(PesVals[s, j], max_val) # Can't be larger than max_val

        #         temp_dot = 0
        #         for temp_state in range(self.nState):
        #             temp_dot+=P_eval[s, j][temp_state]*OptVals[temp_state, j+1]

        #         OptVals[s, j] = temp_expected_reward  + temp_pessimism + temp_dot
        #         OptVals[s, j] = max(OptVals[s, j], min_val)
        #         OptVals[s, j] = min(OptVals[s, j], max_val)

        # return (OptVals, PesVals)

        OptVals = {}
        PesVals = {}

        OptVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        PesVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        test = [tuple(i[1]) for i in list(self.rev_state_dict.items())]
        test = np.array(test)
        
        print("testtt-----------------")
        for i in range(self.epLen):
            j = self.epLen - i - 1
            OptVals[j] = np.zeros(self.nState, dtype=np.float32)
            PesVals[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                OptVals[j][s] = 0
                PesVals[j][s] = 0

                for a in range(self.nAction):
                    print(R.predict(np.array([self.rev_state_dict[s][0], self.rev_state_dict[s][1], a]).reshape(1, -1))[0])
                    max_val = min(self.max_step_reward * (i + 1), self.abs_max_ep_reward) # Smallest max_value
                    min_val = max(self.min_step_reward * (i + 1), -self.abs_max_ep_reward) # Largest min_value
                    final_input = np.concatenate((np.broadcast_to(self.rev_state_dict[s], (self.nState, 2)), np.broadcast_to(np.int64(a),(self.nState, 1)), test), axis = 1)
                    pes_action_contribution = R.predict(np.array([self.rev_state_dict[s][0], self.rev_state_dict[s][1], a]).reshape(1, -1))[0] - pessimism[s, a] + np.dot(P.predict(final_input), PesVals[j + 1])
                    pes_action_contribution = max(pes_action_contribution, min_val) # Can't be lower than min_val.
                    pes_action_contribution = min(pes_action_contribution, max_val) # Can't be larger than max_val.
                    PesVals[j][s] += self.evalpolicy[s, j][a] * pes_action_contribution

                    opt_action_contribution = R.predict(np.array([self.rev_state_dict[s][0], self.rev_state_dict[s][1], a]).reshape(1, -1))[0] + pessimism[s, a] + np.dot(P.predict(final_input), OptVals[j + 1])
                    opt_action_contribution = max(opt_action_contribution, min_val) # Can't be lower than min_val.
                    opt_action_contribution = min(opt_action_contribution, max_val) # Can't be larger than max_val.
                    OptVals[j][s] += self.evalpolicy[s, j][a] * opt_action_contribution

        return (OptVals, PesVals)

                
                



    def update_intervals(self, trajectory_states, R_hat, P_hat):
        '''
        Update intervals via UCBVI.
        '''
        # Output the MAP estimate MDP
        # R_hat, P_hat = self.map_mdp()

        # Pessimistic bonus.
        pessimism = self.gen_bonus(R_hat)

        # uncertainty for each timestep and state
        OptVals_uncertainty, PesVals_uncertainty = self.uncertainty_prop(R_hat, P_hat, pessimism)
        
        # Get shift
        shift = self._get_shift(R_hat, P_hat)

        # Form approximate Q-value estimates
        average_v, total_uncertainty = self.compute_rectifiedVals_evalpolicy(R_hat, P_hat, pessimism, shift, OptVals_uncertainty, PesVals_uncertainty, trajectory_states)
        
        self.total_uncertainty = total_uncertainty
        # self.OptVals = OptVals
        # self.PesVals = PesVals

        # emp_dist = self.behavioral_state_distribution[0]
        # lower_bound = np.dot(emp_dist, PesVals[0])/np.sum(emp_dist)
        # upper_bound = np.dot(emp_dist, OptVals[0])/np.sum(emp_dist)
        self.interval = (average_v - total_uncertainty, average_v+ total_uncertainty)
        # print(self.interval)
    
    def get_interval(self):
        return self.interval[1]

    def get_total_uncertainty(self):
        return self.total_uncertainty