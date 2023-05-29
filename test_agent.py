from collections import deque
from itertools import count

from envs.high_env import HighLevelEnv
from envs.low_env import LowLevelEnv
from neural_networks.high_agent_nn import HighNetwork
from neural_networks.low_agent_nn import LowNetwork
from helpers.graph_gen import GraphGen
from helpers.time_sim import Time_Sim
import torch
import numpy as np

args_ = {
        'n_subs' : 3,
        'n_sub_nodes': 20,
        'sub_ftrs' : 6,
        'max_vnr_nodes' : 10,
        'vnr_ftrs_high':3,
        'vnr_ftrs_low':5,
        'prob_of_link':0.5,
        'max_vnr_not_embed': 1,
        'max_vnr_generated' : 500,
        'episodes': 3000,
        'arrival_rate':0.04,
        'filters_1' : 8,
        'filters_2' : 4,
        'filters_3' : 2,
        'dropout' : 0.5,
        'ntn_neurons' : 16,
        'bins' : 8,
        'histogram' : True,
        'shared_layer_1_n': 32,
        'shared_layer_2_n' : 16,
        'batch_size' : 128,
        'gamma' : 0.99,
        'eps_start': 0.9,
        'eps_end' : 0.005,
        'eps_decay': 1000,
        'tau' : 0.005,
        'lr_high' : 1e-4,
        'lr_low' : 1e-4,
    }


def evaluate(arr_rate, depart_time, resource_size, test_=None):

    args_['arrival_rate'] = arr_rate
    
    gr_gen = GraphGen(n_subs=args_['n_subs'],
                sub_nodes= args_['n_sub_nodes'],
                max_vnr_nodes= args_['max_vnr_nodes'],
                prob_of_link= args_['prob_of_link'],
                vnr_values=resource_size)
    sub_graphs = gr_gen.load_sub_graphs('models/changed_sub_vnr_values')

    acc_high = HighNetwork(args_)
    acc_low = LowNetwork(args_)
    # load the previous models and keep them as evaluating
    acc_high.load_state_dict(torch.load('models/changed_sub_vnr_values/high_agent.pth'))
    acc_high.eval()
    acc_low.load_state_dict(torch.load('models/changed_sub_vnr_values/low_agent.pth'))
    acc_low.eval()
    high_env = HighLevelEnv(sub_graphs, gr_gen, args_['max_vnr_generated'])
    low_env = LowLevelEnv(args_['n_sub_nodes'], args_['n_subs'])
    time_sim = Time_Sim(args_['arrival_rate'], dep_t=depart_time)

    time_sim.reset()
    arr_t = time_sim.ret_arr_time()
    n_vnrs_generated = 0
    n_vnrs_embedded = 0
    score = 0
    cnt_ = 1
    utils = []
    sub_embedded = [0, 0, 0]

    observation = high_env.reset()
    ac_ratio = []

    for step in count():
        
        if n_vnrs_generated > 1000:
            break

        high_env.release_vnrs(step)

        if step >= arr_t:
            n_vnrs_generated += 1
            high_action = acc_high.select_action(observation, train=False)
            # print(high_action)
            embedder_obj = high_env.give_vnr_get_vne_obj(high_action)
            curr_vnr = high_env.get_curr_vnr()
            low_observation = low_env.reset(embedder_obj, high_action)
            performed_actions = []
            for x in range(len(curr_vnr['nodes'])):
                low_action = acc_low.select_action(low_observation, train=False)
                while True:
                    if low_action in performed_actions or low_action is None:
                        low_action = low_env.sample_action()
                    else:
                        break
                performed_actions.append(low_action)
                next_state_low, _, _ = low_env.step(low_action, x)
                low_observation = next_state_low

            embedder_obj = low_env.get_embedder()
            departure_time = time_sim.ret_dep_time(arr_t)
            next_state_high, reward_high, _, fully_embedded = high_env.step(high_action, embedder_obj, n_vnrs_generated, departure_time, cnt_)

            if fully_embedded:
                n_vnrs_embedded += 1
                sub_embedded[high_action] += 1
            score += reward_high
            observation = next_state_high

            arr_t = time_sim.ret_arr_time()
            utils.append(np.mean([high_env.RPs[x].return_avg_util() for x in range(len(high_env.RPs))]))

            ac_ratio.append(n_vnrs_embedded / n_vnrs_generated)
    
    rev_cost_, cum_rev_ = high_env.rev_cost_ratio()
    f = open('result_text_files/results_a3c.txt', 'a')
    f.write('test:{}'.format(test_) + '\n')
    f.write(str('algo:{}'.format('a3c')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write(str('accept_ratio:{}'.format(str(ac_ratio[-1]))) + '\n')
    f.write(str('rev_cost:{}'.format(str(np.mean(rev_cost_)))) + '\n')
    f.write(str('cum_rev:{}'.format(str(np.mean(cum_rev_)))) + '\n\n')
    f.close()

    f = open('result_text_files/util_results_a3c.txt', 'a')
    f.write('test:{}'.format(test_) + '\n')
    f.write(str('algo:{}'.format('a3c')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write('util : {}'.format(utils))
    f.write('\n\n')
    f.close()

    f = open('result_text_files/vnr_dist_a3c.txt', 'a')
    f.write(str('algo:{}'.format('a3c')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write('vnr distribution : {}'.format(sub_embedded))
    f.write('\n\n')
    f.close()

    return sub_embedded

arr_rates = [0.04 * x for x in range(1, 6)]
depart_times = [400+ (x * 100) for x in range(1, 6)]
resource_sizes = [
    [[3, 6], [2,4], [2,4,8], [32, 48]],
    [[3, 6], [4,6], [4,6,10], [36,52]],
    [[3, 6], [6,8], [6,8,12], [40,56]],
    [[3, 6], [8,10], [8,10,14], [44,60]],
    [[3, 6], [10,12], [10,12,16], [48,64]]
]

values = [arr_rates, depart_times, resource_sizes]
i = 0

for value in values:
    arr_r = 0.04
    dep_t = 600
    res_size = [[3, 6], [2,4], [2,4,8], [32, 48]]
    for val in value:    
        if i == 0:
            arr_r = val
            test = 'arr_rate'
        elif i == 1:
            dep_t = val
            test = 'dept_time'           
        else:
            res_size = val
            test = 'vnr_size'
        evaluate(arr_r, dep_t, res_size, test_=test)
    i += 1