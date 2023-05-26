import sys
from collections import deque
from itertools import count

from envs.high_env import HighLevelEnv
from envs.low_env import LowLevelEnv
from neural_networks.high_agent_nn import HighNetwork
from neural_networks.low_agent_nn import LowNetwork
from helpers.graph_gen import GraphGen
from helpers.time_sim import Time_Sim
from helpers.shared_adam import SharedAdam
import time
import os

import torch
import torch.multiprocessing as mp

class ActorCriticAgent(mp.Process):
    def __init__(self, g_high_acc_net, g_low_acc_net, opt_high, opt_low, name, global_ep_idx, gr_gen, sub_graphs, args=None) -> None:
        super(ActorCriticAgent, self).__init__()
        self.args = args
        self.gr_gen = gr_gen
        self.sub_graphs = sub_graphs
        self.high_env = HighLevelEnv(self.sub_graphs, self.gr_gen, args['max_vnr_generated'])
        self.low_env = LowLevelEnv(args['n_sub_nodes'], args['n_subs'])
        self.time_sim = Time_Sim(args['arrival_rate'])

        self.local_high_acc_net = HighNetwork(args)
        self.local_low_acc_net = LowNetwork(args)
        self.global_high_acc = g_high_acc_net
        self.global_low_acc = g_low_acc_net
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.opt_high = opt_high
        self.opt_low = opt_low
        
    
    def run(self):

        while self.episode_idx.value < self.args['episodes']:
            self.time_sim.reset()
            arr_t = self.time_sim.ret_arr_time()
            n_vnrs_generated = 0
            n_vnrs_embedded = 0
            self.rew_buffer = deque([0.0], maxlen=1000)
            done_high = False
            score = 0
            prev_option = None
            cnt_ = 1

            # this function creates a new vnr and stores it as a member variable 
            # in high environment object and resets all the sub graphs to inital state
            observation = self.high_env.reset()
            self.local_high_acc_net.clear_memory()
            
            for step in count():
                
                self.high_env.release_vnrs(step)

                if step >= arr_t:
                    n_vnrs_generated += 1
                    high_action = self.local_high_acc_net.select_action(observation)
                    if prev_option == high_action:
                        cnt_ += 1
                    else:
                        cnt_ = 1
                    prev_option = high_action
                    # have to choose the embedder using high env and then pass to low lev env to embed the vnr
                    # the vnr is a member variable of the high env itself so no need to pass
                    embedder_obj = self.high_env.give_vnr_get_vne_obj(high_action)
                    curr_vnr = self.high_env.get_curr_vnr()
                    low_observation = self.low_env.reset(embedder_obj, high_action)
                    self.local_low_acc_net.clear_memory()
                    performed_actions = []
                    # we must do this because we want each of the state reward and action for low agent
                    for x in range(len(curr_vnr['nodes'])):
                        low_action = self.local_low_acc_net.select_action(low_observation)
                        while True:
                            if low_action in performed_actions or low_action is None:
                                low_action = self.low_env.sample_action()
                            else:
                                break
                        performed_actions.append(low_action)
                        next_state_low, reward_low, done_low = self.low_env.step(low_action, x)

                        # store and optimize
                        self.local_low_acc_net.remember(low_observation, low_action, reward_low)
                        
                        low_observation = next_state_low
                        if done_low:
                            loss = self.local_low_acc_net.calc_loss(done_low)
                            self.opt_low.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_value_(self.local_low_acc_net.parameters(), 100)
                            for local_param, global_param in zip(self.local_low_acc_net.parameters(), self.global_low_acc.parameters()):
                                global_param._grad = local_param.grad
                            self.opt_low.step()
                            self.local_low_acc_net.load_state_dict(self.global_low_acc.state_dict())
                            self.local_low_acc_net.clear_memory()
                            break
                        
                    embedder_obj = self.low_env.get_embedder()
                    departure_time = self.time_sim.ret_dep_time(arr_t)
                    # embeds link, gets mapping, changes sub, stores the mapp, replaces the previous
                    # embedder object in the list with the new one
                    vnrs_not_embedded = n_vnrs_generated - n_vnrs_embedded
                    next_state_high, reward_high, done_high, fully_embedded = self.high_env.step(high_action, embedder_obj, n_vnrs_generated, departure_time, cnt_)

                    if fully_embedded:
                        n_vnrs_embedded += 1

                    # store and optimize
                    self.local_high_acc_net.remember(observation, high_action, reward_high)
                    score += reward_high

                    observation = next_state_high
                    if done_high:
                        loss = self.local_high_acc_net.calc_loss(done_low)
                        self.opt_high.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(self.local_high_acc_net.parameters(), 100)
                        for local_param, global_param in zip(self.local_high_acc_net.parameters(), self.global_high_acc.parameters()):
                            global_param._grad = local_param.grad
                            
                        self.opt_high.step()
                        self.local_high_acc_net.load_state_dict(self.global_high_acc.state_dict())
                        self.local_high_acc_net.clear_memory()
                        break
                    
                    arr_t = self.time_sim.ret_arr_time()
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score , 'vnrs_embedded : ', n_vnrs_embedded, 'vnrs_gen : ' , n_vnrs_generated)

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
        'arrival_rate':0.2,
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
        'lr_high' : 1e-3,
        'lr_low' : 1e-4,
    }

if __name__ == '__main__':

    gr_gen = GraphGen(n_subs=args_['n_subs'],
                    sub_nodes= args_['n_sub_nodes'],
                    max_vnr_nodes= args_['max_vnr_nodes'],
                    prob_of_link= args_['prob_of_link'])
    subs,df_to_save = gr_gen.make_cnst_sub_graphs()

    global_acc_high = HighNetwork(args_)
    global_acc_high.share_memory()
    global_acc_low = LowNetwork(args_)
    global_acc_low.share_memory()

    optim_high = SharedAdam(global_acc_high.parameters(), lr=args_['lr_high'], 
                        betas=(0.92, 0.999))
    optim_low = SharedAdam(global_acc_low.parameters(), lr=args_['lr_low'], 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [ActorCriticAgent(global_acc_high,
                                global_acc_low,
                                optim_high,
                                optim_low,
                                name=i,
                                global_ep_idx=global_ep,
                                gr_gen=gr_gen,
                                sub_graphs = subs,
                                args=args_) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]

    foldername_ = 'changed_sub_vnr_values'
    os.mkdir('models/' + foldername_)
    torch.save(global_acc_high.state_dict(), 'models/'+ foldername_ +'/high_agent.pth')
    torch.save(global_acc_low.state_dict(), 'models/'+ foldername_ +'/low_agent.pth')
    df_to_save.to_csv('models/'+ foldername_ +'/sub_graphs_used.csv', index=False)



    


