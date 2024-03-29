'''
Credit:
- This python file is a derivative work of https://github.com/iosband/TabulaRL
- The main changes being made are to facilitate offlineRL.

FiniteHorizonTabularAgent has the following methods:
- update_obs:
    - Input:
        - oldState - int
        - action - int
        - reward - double
        - newState - int
        - pContinue - 0/1
        - h - int - time within episode (not used)
    - Updates self.R_prior and self.P_prior.
- egreedy:
    - Input:
        - state - int - current state
        - timestep - int - timestep *within* episode
    - Returns greedy action based on maximizing the current Q-value (self.qVals[state, timestep]).
- pick_action:
    - Input:
        - state - int - current state
        - timestep - int - timestep *within* episode
    - Returns action according to egreedy.
- sample_mdp:
    - No input.
    - Returns R_samp, and P_samp from prior.
- map_mdp:
    - No input.
    - Returns R_hat, P_hat which is the center of the current prior.
- compute_qVals:
    - Computes value iteration. [Only need to compute H steps for the finite horizon case]
    - Input:
        - R - R[s,a] = mean rewards
        - P - P[s,a] = probability vector of transitions
    - Returns:
        - qVals - qVals[state, timestep] is vector of Q values for each action
        - qMax - qMax[timestep] is the vector of optimal values at timestep
- compute_qVals_opt:
    - Computes optimistic value iteration.
    - Input:
        - R - R[s,a] = mean rewards
        - P - P[s,a] = probability vector of transitions
        - R_bonus - R_bonus[s,a] = bonus for rewards
        - P_bonus - P_bonus[s,a] = bonus for transitions
    - Returns:
        - qVals - qVals[state, timestep] is vector of Q values for each action
        - qMax - qMax[timestep] is the vector of optimal values at timestep
    - Details on the two bonuses:
        - R_bonus[s, a] is added to the rewards.
        - P_bonus[s, a]*i is added for step j = self.epLen - i - 1 (additional -1 is to manage python indexing starting from 0).
- compute_qVals_EVI:
    - Computes extended value iteration: https://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf
    - pOpt finds a transition that maximizes the next step value.
    - Input:
        - R - R[s,a] = mean rewards
        - P - P[s,a] = probability vector of transitions
        - R_slack - R_slack[s,a] = slack for rewards
        - P_slack - P_slack[s,a] = slack for transitions
    - Returns:
        - qVals - qVals[state, timestep] is vector of Q values for each action
        - qMax - qMax[timestep] is the vector of optimal values at timestep
'''

import numpy as np

class FiniteHorizonTabularAgent():
    '''
    Simple tabular offlineRL agent.

    Child agents will mainly implement:
        update_policy (updates the policy used by egreedy and pick_action)
        And helper functions (e.g. compute_QVals, get_bonus, etc)

    Important internal representation is given by qVals and qMax.
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep

    '''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., **kwargs):
        '''
        Tabular episodic learner for time-homoegenous MDP.
        Must be used together with true state feature extractor.

        Args:
            nState - int - number of states
            nAction - int - number of actions
            alpha0 - prior weight for uniform Dirichlet
            mu0 - prior mean rewards
            tau0 - precision of prior mean rewards
            tau - precision of reward noise

        Returns:
            tabular learner, to be inherited from
        '''
        # Instantiate the Bayes learner
        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau

        self.qVals = {}
        self.qMax = {}

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}
        self.behavioral_state_distribution = np.zeros([self.epLen+1, self.nState], dtype=np.float32)

        for state in range(nState):
            for action in range(nAction):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.nState, dtype=np.float32))

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
        self.behavioral_state_distribution[h-1][oldState] += 1

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

    def egreedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        Q = self.qVals[state, timestep]
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])

        return action

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep)
        return action

    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

        return R_samp, P_samp

    def map_mdp(self):
        '''
        Returns the maximum a posteriori MDP from the posterior.

        Args:
            NULL

        Returns:
            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
        '''
        R_hat = {}
        P_hat = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_hat[s, a] = self.R_prior[s, a][0]
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_hat, P_hat

    def compute_qVals(self, R, P):
        '''
        Compute the Q values for a given R, P estimates

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
                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_opt(self, R, P, R_bonus, P_bonus):
        '''
        Compute the Q values for a given R, P estimates + R/P bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_bonus - R_bonus[s,a] = bonus for rewards
            P_bonus - P_bonus[s,a] = bonus for transitions

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
                    qVals[s, j][a] = (R[s, a] + R_bonus[s, a]
                                      + np.dot(P[s, a], qMax[j + 1])
                                      + P_bonus[s, a] * i)
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_EVI(self, R, P, R_slack, P_slack):
        '''
        Compute the Q values for a given R, P by extended value iteration

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_slack - R_slack[s,a] = slack for rewards
            P_slack - P_slack[s,a] = slack for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
                # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    rOpt = R[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = P[s, a]
                    if pOpt[pInd[self.nState - 1]] + P_slack[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.nState)
                        pOpt[pInd[self.nState - 1]] = 1
                    else:
                        pOpt[pInd[self.nState - 1]] += P_slack[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_pes(self, R, P, pessimism, is_positive = True):
        '''
        Currently unused as PVIAgent has it's own implementation.
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
                    if is_positive:
                        qVals[s, j][a] = (max(R[s, a] - pessimism[s, a] + np.dot(P[s, a], qMax[j + 1]), 0))
                    else:
                        qVals[s, j][a] = (R[s, a] - pessimism[s, a] + np.dot(P[s, a], qMax[j + 1]))
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax
    
    def _update_evalpolicy(self, policy):
        self.evalpolicy = policy