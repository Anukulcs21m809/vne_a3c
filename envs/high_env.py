from networkx.readwrite import json_graph
from networkx.algorithms import shortest_paths
import math
import numpy as np
import copy
import random
import sys

sys.path.insert(1, "../helpers")
from helpers.featurizer import Featurizer
from helpers.network_embedder import VNE_on_sub
from helpers.graph_gen import GraphGen

# we use the same graph generator from the main program for all the environments
class HighLevelEnv:
    def __init__(self, sub_graphs, gr_gen, max_vnr_generated) -> None:
        self.sub_graphs = sub_graphs
        self.RPs = [VNE_on_sub(self.sub_graphs[i]) for i in range(len(self.sub_graphs))]
        self.gr_gen = gr_gen
        self.curr_vnr = self.gr_gen.gen_rnd_vnr()
        self.max_vnr_generated = max_vnr_generated
        self.reward_scaling = 10
        self.revenues = [0,0,0]
        self.costs = [0,0,0]
        self.cum_rev = 0

    @property
    def action_shape(self):
        return len(self.sub_graphs)
    
    def sample_action(self):
        return random.randint(0, self.action_shape - 1)
    
    def encode_graphs(self):
        state_ = []
        for obj in self.RPs:
            state_.append(Featurizer.make_data_obj(obj.curr_state, sub=1, high=1))
        state_.append(Featurizer.make_data_obj(self.curr_vnr, sub=0, high=1))
        return state_

    # reset is only done in the beginning of an episode
    def reset(self):
        for x in range(len(self.RPs)):
            self.RPs[x].reset_to_original()
        self.curr_vnr = self.gr_gen.gen_rnd_vnr()
        return self.encode_graphs()
    
    def release_vnrs(self, time_step):
        for x in range(len(self.RPs)):
            revenues, costs = self.RPs[x].release_resources(time_step)
            self.revenues[x] -= revenues
            self.costs[x] -= costs

    # this function will provide the curr vnr to the embedding object and then get the object back
    # this vne obj will be used by the low agent
    def give_vnr_get_vne_obj(self, action):
        self.RPs[action].receive_vnr(self.curr_vnr)
        self.RPs[action].reset_map()
        return self.RPs[action]
    
    def get_curr_vnr(self):
        return self.curr_vnr
    
    def get_reward(self, embedded, mapp, cnt, action):
        r_h_e = self.reward_scaling if embedded else -1 * self.reward_scaling
        if mapp == None or len(mapp['link_ind']) < 1:
            r_h_u = 1
        else:
            r_h_u = 0
            for link in mapp['link_ind']:
                avg_resource = 0
                for ind_ in link:
                    sub_link = self.RPs[action].curr_state['links'][ind_]
                    avg_resource += (sub_link['bw']/ sub_link['band_max'])
                r_h_u += (avg_resource / len(link))
            r_h_u /= len(mapp['link_ind'])
        
        if not embedded:
            r_h_rc = 1 # this means that we are making the reward more negative
        else:
            r_h_rc = 0
            rev_change = 0
            cost_change = 0
            for cpu_mem in mapp['cpu_mem']:
                rev_change = cpu_mem[0] + cpu_mem[1]
            cost_change = copy.deepcopy(rev_change)
            for u in range(len(mapp['paths'])):
                rev_change += mapp['bw'][u]
                cost_change += (mapp['bw'][u] * (len(mapp['paths'][u]) - 1))
            r_h_rc = (rev_change / cost_change)
        
        r_c = 1 / (math.pow((cnt - 1), 2) + 1)

        final_rew = (r_h_e * r_h_u * r_h_rc) * r_c
        return final_rew 
    
    # this embedd obj is received after node embedding by the low level agent
    # all the functions only execute if the nodes and the links are embedded 
    # embed link function only runs if all the nodes have been embedded by the low agent
    def step(self, action, embedd_obj, vnrs_gen, vnr_dept_time, cnt):
        embedded = embedd_obj.embed_link()
        mapp, fully_embedded = embedd_obj.get_mapping()
        reward = self.get_reward(embedded, mapp, cnt, action)

        rev, cost = embedd_obj.return_rev_cost(mapp)
        self.revenues[action] += rev
        self.costs[action] += cost
        self.cum_rev += rev

        embedd_obj.change_sub()
        embedd_obj.store_map(vnr_dept_time)

        self.RPs[action] = embedd_obj
        # we need to change the vnr for the next state
        self.curr_vnr = self.gr_gen.gen_rnd_vnr()

        # for now done is True if the max time is reached otherwise we could also make it true if a VNR is not embedded
        # or some number of VNRs are not embedded
        done = True if vnrs_gen > self.max_vnr_generated else False
        next_state = self.encode_graphs() #if not done else None
        
        
        return next_state, reward, done, fully_embedded

    def rev_cost_ratio(self):
        tot_rev_ratio = np.mean([0 if c == 0 else r / c for r, c in zip(self.revenues, self.costs)])
        return tot_rev_ratio, self.cum_rev
    
    
