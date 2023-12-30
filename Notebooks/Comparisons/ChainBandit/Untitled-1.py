# %%
# imports
from OfflineSRL.MDPDataset.old_dataset import *
from OfflineSRL.MDP.old_MDP import MDP
from OfflineSRL.MDP.ChainBandit import ChainBanditMDP
from OfflineSRL.MDP.ChainBanditState import ChainBanditState
from OfflineSRL.BPolicy.ChainBanditPolicy import ChainBanditPolicy
from OfflineSRL.OfflineLearners.offlineLearners import VI, PVI, SPVI, PesBandit
from OfflineSRL.OfflineEvaluators.offlineEvaluator import SPVIEval, StandardPesEval, SelectivePesEval
from statistics import mean, pstdev
import math
import copy
import numpy as np

# %%
def get_dataset(horizon = 3, neps = 50, third_action_prob = 0.2, num_states =10):
    # Initialize MDP and Policy
    mdp = ChainBanditMDP(num_states = num_states)
    policy = ChainBanditPolicy(mdp, third_action_prob = third_action_prob)
    
    # Generate data
    observations = []
    actions = []
    rewards = []
    terminals = []
    for eps in range(neps):
        for timestep in range(horizon+2):
            # Get state.
            # Add state to list.
            cur_state = copy.deepcopy(mdp.cur_state)
            
            observations.append(copy.deepcopy(cur_state.num_list))

            # Get action
            # Add action to list
            cur_action = policy._get_action(state = cur_state)
            
            actions.append(copy.deepcopy(cur_action))

            # Execute action
            reward, next_state = mdp.execute_agent_action(cur_action)
            # Add reward
            rewards.append(copy.deepcopy(reward))

            terminals.append(0)
        mdp.reset()
        terminals[-1] = 1
        
    # Convert to MDPDataset format
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    
    return observations, policy, dataset

# %%
def evaluate_learner(option, observations, policy, dataset, horizon, neps = 5000, num_states =10, eval_policy=None):
    max_step_reward = 1
    abs_max_ep_reward = 1
    min_step_reward = 0
    # if option == "VI":
    #     vi = VI(name = "vi", states = observations, actions = policy.actions, epLen = horizon,is_eval = False)
    # if option == "PVI":
    #     vi = PVI(name = "pvi", states = observations, actions = policy.actions, epLen = horizon, is_eval = False,
    #              max_step_reward = max_step_reward, min_step_reward = min_step_reward, abs_max_ep_reward = abs_max_ep_reward)
    # if option == "SPVI":
    #     vi = SPVI(name = "spvi", states = observations, actions = policy.actions, epLen = horizon, bpolicy = policy, is_eval = False,
    #               max_step_reward = max_step_reward, min_step_reward = min_step_reward, abs_max_ep_reward = abs_max_ep_reward)
    # if option == "PSL":
    #     vi = PesBandit(name = "psl", states = observations, actions = policy.actions, epLen = horizon, is_eval = False)

    # vi.fit(dataset)
    # eval_policy = {}
    # # eval_policy
    # print(vi.n_states)
    # print("number of states")
    # for timestep in range(horizon):
    #     for s in range(vi.n_states):
    #         eval_policy[s, timestep] = vi.agent.action_prob(s, timestep)
  

    uncertainty_vec_spvi = []
    uncertainty_vec_pvi = []
    # alphas = [0, 0.2, 0.4, 0.6, 0.8]
    # neps = [10000, 20000, 50000]
    # temp = [0.1,0.1,0.8]

    final_policy = {}
    # for timestep in range(horizon):
    #     for s in range(vi.n_states):

    #         final_policy[s, timestep] = eval_policy[s, timestep]*alpha + (1-alpha)*(np.array(policy._get_probs(s, timestep)))
    #         if s == 10 and timestep == 2:
    #             print("insdie the specific if")
                
    #         # final_policy[s, timestep] = (np.array(policy._get_probs(s, timestep)))

    for timestep in range(horizon):
        for s in range(2*num_states+1):
            final_policy[s, timestep] = eval_policy

    pvi_eval = StandardPesEval(name = "pvi", states = num_states, actions=policy.actions, epLen=horizon, evalpolicy=final_policy, is_eval = True, is_finite = True)
    vi_eval = SPVIEval(name = "spvi", states = num_states, actions = policy.actions, epLen = horizon, bpolicy = policy,evalpolicy=final_policy,data_splitting=0, delta=0.9, nTrajectories=neps, epsilon1=0.0001, epsilon2=0.000001, epsilon3=0.00001,
                max_step_reward = max_step_reward, min_step_reward = min_step_reward, abs_max_ep_reward = abs_max_ep_reward, is_eval = True, is_finite = True)

    pvi_new_eval = SelectivePesEval(name = "pvi", states = num_states, actions=policy.actions, epLen=horizon, evalpolicy=final_policy,bpolicy = policy, is_eval = True, is_finite = True)


    pvi_eval.fit(dataset)
    # vi_eval.fit(dataset)
    
    pvi_new_eval.fit(dataset)
    # uncertainty_vec_pvi.append(pvi_eval.agent.get_interval())
    # uncertainty_vec_spvi.append(vi_eval.agent.get_interval())
        
    return (pvi_eval.agent.get_interval(), pvi_new_eval.agent.get_interval())
    # mdp = ChainBanditMDP(num_states = horizon)
    # viobservations = []
    # viactions = []
    # virewards = []
    # viterminals = []
    # for eps in range(neps):
    #     for timestep in range(horizon):
    #         # Get state.
    #         # Add state to list.
    #         cur_state = copy.deepcopy(mdp.cur_state)
    #         viobservations.append(copy.deepcopy(cur_state.num_list))

    #         # Get action
    #         # Add action to list
    #         cur_action = vi.act(copy.deepcopy(cur_state.num_list), timestep)
    #         viactions.append(copy.deepcopy(cur_action))

    #         # Execute action
    #         reward, next_state = mdp.execute_agent_action(cur_action)
    #         # Add reward
    #         virewards.append(copy.deepcopy(reward))

    #         viterminals.append(0)
    #     mdp.reset()
    #     viterminals[-1] = 1
    # return np.sum(np.array(virewards))/neps

# %%
rew_dict = {}
option_list = ["PSL","PVI","SPVI"]
# for option in option_list:
#     rew_dict[option] = {}
n_runs = 1
horizon = [5,10,20,30]
neps_list = [2000,4000,6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

pvi_vec = []
spvi_vec = []
error_1_pvi = []
errror_2_spvi = []
num_states = 5


policies = [[0.02,0.49,0.49], [0.1,0.40,0.40], [0.20,0.40,0.40], [0.30,0.35,0.35], [0.40,0.30,0.30], [0.60,0.20,0.20], [0.80, 0.10, 0.10]]
# for neps in neps_list:
#     print(neps)
#     # for option in option_list:
#     #     rew_dict[option][neps] = []
#     rew_dict[neps] = []
#     for run in range(n_runs):
#         observations, policy, dataset = get_dataset(horizon = horizon, neps = neps, third_action_prob = 0.8)
#         # for option in option_list:
#         rew_dict[neps].append(evaluate_learner("SPVI", copy.deepcopy(observations), policy, dataset, horizon, neps))
            #print(option)
            #print(option, neps, evaluate_learner(option, copy.deepcopy(observations), policy, dataset, horizon))

for h in policies:
    print("this is the policy being followed")
    print(h)
    # observations, policy, dataset = get_dataset(horizon = 10, neps = h, third_action_prob = 0.01, num_states = num_states)
    obs_1 = []
    obs_2 = []
    for times in range(1):
        observations, policy, dataset = get_dataset(horizon = 6, neps = 2000, third_action_prob = 0.02, num_states = num_states)
        print(dataset)
        l, p = evaluate_learner("SPVI", copy.deepcopy(observations), policy, dataset, 6, 2000, num_states,h)  
        pvi_vec.append(l)
        spvi_vec.append(p)
    # mean_1 = mean(obs_1)
    # mean_2 = mean(obs_2)
    # error_1 = pstdev(obs_1) / math.sqrt(20)
    # error_2 = pstdev(obs_2) / math.sqrt(20)
    # error_1_pvi.append(error_1)
    # errror_2_spvi.append(error_2)
    # pvi_vec.append(mean_1)
    # spvi_vec.append(mean_2)


# %%
# rew = {}
# err = {}
# bounds = []
# # for option in option_list:
# #     rew[option] = []
# #     err[option] = []
# for neps in neps_list:
#     # for option in option_list:
#         bounds.append(np.mean(rew_dict[neps]))
        # err[option].append(np.std(rew_dict[option][neps])/np.sqrt(n_runs))

# %%
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
# %matplotlib inline

fig, ax = plt.subplots()
# for option in option_list:
x = [0.02, 0.1, 0.20, 0.30, 0.40, 0.60, 0.80]
y = pvi_vec
y1 = spvi_vec
# yerr = err[option]
# ax.errorbar(x, y)
#             # yerr=yerr,
#             # fmt='-o', label = option)

plt.plot(x, y, **{'marker':'o'})

plt.plot(x,y1, **{'marker' : 'o'})
plt.xlabel("Third action probability")
plt.ylabel("Interval width")
plt.legend(["Standard Pessimism", "Selective Pessimism"])
plt.title("Chain Bandit: horizon=10")
print(x)
print(y)
# ax.set_xlabel('Number of training episodes')
# ax.set_ylabel('Test reward')
# ax.set_title('Chain Bandit: horizon = '+str(horizon))
# plt.legend()

# plt.savefig('chainbandit-h=3-suopt=0.8.png')
plt.show()

# %%
# fig, ax = plt.subplots()
# for option in option_list:
#     x = neps_list
#     y = np.array(rew[option])
#     yerr = np.array(err[option])
#     ax.plot(x, y, '-o', label = option)
#     ax.fill_between(x, y-yerr, y+yerr)
#     #ax.errorbar(x, y,
#     #            yerr=yerr,
#     #            fmt='-o', label = option)


# ax.set_xlabel('Number of training episodes')
# ax.set_ylabel('Test reward')
# ax.set_title('Chain Bandit: horizon = '+str(horizon))
# plt.legend()

# #x = np.linspace(0, 30, 30)
# #y = np.sin(x/6*np.pi)
# #error = np.random.normal(0.1, 0.02, size=y.shape)
# #y += np.random.normal(0, 0.1, size=y.shape)

# #plt.plot(x, y, 'k-')
# #plt.fill_between(x, y-error, y+error)
# plt.show()

# %%



