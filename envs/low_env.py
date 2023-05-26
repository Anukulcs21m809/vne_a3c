# a global optimizer is initialized for two kinds of agents (actor and the critic)
# gradient update is done on the global optimizer only

import random
import sys
import json
import pandas as pd
import copy
import numpy as np

sys.path.insert(1, "../helpers")
from helpers.featurizer import Featurizer

# needs to receive a network embedder from the high level agent so that it can embed the nodes

class LowLevelEnv:
    # pass the number of substrate nodes
    def __init__(self, no_of_sub_nodes, n_subs) -> None:
        self.no_of_sub_nodes = no_of_sub_nodes
        self.rew_scaling_factor = 1
        self.repeat_matrix = np.zeros((n_subs, self.action_shape))
    
    @property
    def action_shape(self):
        return self.no_of_sub_nodes
    
    def sample_action(self):
        return random.randint(0, self.action_shape - 1)
    
    # must receive an embedder object that contains the substrate graph for every vnr that we process
    def reset(self, embedder, embedder_ind):
        self.curr_vnr = embedder.curr_eval_vnr
        self.embdder_ind = embedder_ind
        self.resource_allocator = embedder
        # self.resource_allocator.reset_map()
        init_state_ = []
        init_state_.append(Featurizer.make_data_obj(self.resource_allocator.curr_state, sub=1, high=0))
        # vnr state is constant, only the flags need to change
        self.vnr_data_obj = Featurizer.make_data_obj(self.curr_vnr, sub=0, high=0, initial=True)
        init_state_.append(self.vnr_data_obj)
        return init_state_
    
    # encodes the node-embedded-sub-graph and the vnr graph to data objects that will be used as states
    # we use the same vnr data obj with the flags changed for each vnr node
    def encode_graphs(self, temp_sub, vnr_node_ind, embedded, final):
        sub_encoding = Featurizer.make_data_obj(temp_sub, sub=1, high=0)
        self.vnr_data_obj = Featurizer.change_flags(self.vnr_data_obj, vnr_node_ind, embedded, final=final)
        state_ = [sub_encoding, self.vnr_data_obj]
        return state_
    
    def get_reward(self, embedded, ind_vnr_, sub_node):
        mult = 1 if embedded else -1
        r_l_e = mult * self.rew_scaling_factor * (1 / (ind_vnr_ + 1))
        r_l_u = (sub_node['cpu'] + sub_node['mem']) / (sub_node['cpu_max'] + sub_node['mem_max'])
        final_rew = r_l_e * r_l_u * (1 / (self.repeat_matrix[self.embdder_ind][ind_vnr_] + 1))
        return final_rew
    
    def step(self, action, vnr_node_ind):
        temp_sub, embeddable = self.resource_allocator.embed_node(action, vnr_node_ind)
        done = True if vnr_node_ind == (len(self.curr_vnr['nodes'])-1) else False 
        next_state = self.encode_graphs(temp_sub, vnr_node_ind, embeddable, done)
        reward = self.get_reward(embeddable, vnr_node_ind, self.resource_allocator.curr_state['nodes'][action])

        return next_state, reward, done
    
    # we call this from the main program , this embedder object will be used by the high level agent
    # the current map in this object might have a partial embedding or a full embedding of the nodes
    # the link can only be embedded if all the nodes are embedded otherswise it returns a partial mapp
    # this is all handled by the embedder object itself
    def get_embedder(self):
        return self.resource_allocator
        