import copy
from networkx.readwrite import json_graph
from networkx.algorithms import shortest_paths
import sys
import numpy as np


# this class will be used separately by all the substrate graphs to embed the VNRs
# both the low level and high level agent uses the functions of this class
class VNE_on_sub:
    def __init__(self, sub_graph) -> None:
        self.sub_graph_original = sub_graph
        self.curr_state = copy.deepcopy(self.sub_graph_original)    
        self.map_skel = {'sub':None, 'vnr_node_ind':[], 'sub_node_ind':[], 'cpu_mem':[], 'link_ind':[], 'bw':[], 'paths':[], 'dep_t':None}
        self.all_mappings = []
        self.curr_map = None
        self.curr_eval_vnr = None
    
    def reset_to_original(self):
        self.curr_state = copy.deepcopy(self.sub_graph_original)
        self.all_mappings = []
    
    def receive_vnr(self, vnr=None):
        if vnr is not None:
            self.curr_eval_vnr = vnr
        else:
            raise ValueError
    
    # call when we need the map to reset to initial values for each vnr
    def reset_map(self):
        self.curr_map = copy.deepcopy(self.map_skel)

    # can only be called by embed node so that we can get the state of current substrate for each node embedding
    def temp_sub_change(self):
        temp_sub = copy.deepcopy(self.curr_state)
        if len(self.curr_map['vnr_node_ind']) != 0 and len(self.curr_map['sub_node_ind']) != 0:
            # just take the recent mapping and change the substrate
            for x in range(len(self.curr_map['sub_node_ind'])):
                temp_sub['nodes'][self.curr_map['sub_node_ind'][x]]['cpu'] -= self.curr_map['cpu_mem'][x][0]
                temp_sub['nodes'][self.curr_map['sub_node_ind'][x]]['mem'] -= self.curr_map['cpu_mem'][x][1]
        return temp_sub

    # first function to call by low level agent to embed each node indexed by the vnr_node_ind
    def embed_node(self, sub_node_ind, vnr_node_ind):
        temp_sub = self.temp_sub_change()

        sub_node = temp_sub['nodes'][sub_node_ind]
        vnr_node = self.curr_eval_vnr['nodes'][vnr_node_ind]

        embeddable = False
        if(sub_node['cpu'] >= 0 and sub_node['mem'] >= 0):
            if(sub_node['cpu'] >= vnr_node['cpu']) and (sub_node['mem'] >= vnr_node['mem']):
                embeddable = True
                self.curr_map['vnr_node_ind'].append(vnr_node_ind)
                self.curr_map['sub_node_ind'].append(sub_node_ind)
                self.curr_map['cpu_mem'].append((vnr_node['cpu'], vnr_node['mem']))
        
        # this is done to obtain a new state if the embedding has taken place
        # otherwise it returns the same old state
        temp_sub = self.temp_sub_change()
        return temp_sub, embeddable

    # this function is only called by the embed link function
    def check_node_embedded(self):
        if len(self.curr_map['vnr_node_ind']) == len(self.curr_eval_vnr['nodes']):
            return True
        else:
            return False
    
    # second function to be called by the high level agent to embed the links only if all the nodes are embedded
    def embed_link(self):
        link_embedded = False
        if self.check_node_embedded():
            nx_sub = json_graph.node_link_graph(self.curr_state)
            reserved_paths = {}

            if len(self.curr_map['vnr_node_ind']) == 0 or len(self.curr_map['sub_node_ind']) == 0:
                return link_embedded, self.curr_map

            for v_link in self.curr_eval_vnr['links']:
                v_s_ind = v_link['source']
                v_t_ind = v_link['target']
                bw = v_link['bw']

                src = self.curr_map['sub_node_ind'][self.curr_map['vnr_node_ind'].index(v_s_ind)]
                tar = self.curr_map['sub_node_ind'][self.curr_map['vnr_node_ind'].index(v_t_ind)]

                # since the substrate is a connected graph, we assume that path exists for each pair of nodes
                path_ = shortest_paths.astar_path(nx_sub, src, tar)

                ind_of_links = []
                for x in range(len(path_) - 1):
                    i = 0
                    index_of_link = None
                    for link in self.curr_state['links']:
                        if((link['source'] == path_[x] and link['target'] == path_[x+1]) or (link['source'] == path_[x+1] and link['target'] == path_[x])):
                            index_of_link = i
                            break
                        i += 1
                    
                    if index_of_link is not None and index_of_link not in reserved_paths.keys():
                        reserved_paths[index_of_link] = self.curr_state['links'][index_of_link]['bw']

                    if index_of_link is not None and reserved_paths[index_of_link] >= 0 and reserved_paths[index_of_link] >= bw:
                        ind_of_links.append(index_of_link)
                        reserved_paths[index_of_link] -= bw
                        link_embedded = True
                    else:
                        link_embedded = False
                        break
                if link_embedded == False:
                    break
                else:
                    self.curr_map['link_ind'].append(ind_of_links)
                    self.curr_map['paths'].append(path_)
                    self.curr_map['bw'].append(bw)

        return link_embedded
    
    def check_link_embedded(self):
        if len(self.curr_eval_vnr['links'])  == len(self.curr_map['paths']):
            return True
        else:
            return False    
    
    # this function returns a mapping only if all the nodes and links are fully embedded
    # and a bool value to indicate if the embedding is successful
    def get_mapping(self):
        if self.check_node_embedded() and self.check_link_embedded():
            return self.curr_map , True
        else:
            return self.map_skel , False

    # only if the nodes and links are embedded
    # change the substrate and then immedaitely add the mapping 
    def change_sub(self):
        if self.check_link_embedded() and self.check_node_embedded():
            for x in range(len(self.curr_map['sub_node_ind'])):
                self.curr_state['nodes'][self.curr_map['sub_node_ind'][x]]['cpu'] -= self.curr_map['cpu_mem'][x][0]
                self.curr_state['nodes'][self.curr_map['sub_node_ind'][x]]['mem'] -= self.curr_map['cpu_mem'][x][1]
            
            for x in range(len(self.curr_map['link_ind'])):
                for y in self.curr_map['link_ind'][x]:
                    self.curr_state['links'][y]['bw'] -= self.curr_map['bw'][x]
        return self.curr_state

    # takes the departure time from the high level agent and then assigns to the map and then stores it
    def store_map(self, dep_time):
        if self.check_link_embedded() and self.check_node_embedded():
            self.curr_map['dep_t'] = dep_time
            self.all_mappings.append((dep_time, self.curr_map))
            self.all_mappings.sort(key = lambda i : i[0])

    # just prvide the step number from the main program
    def release_resources(self, step):
        exceeded = True
        resources = []
        revenues = 0
        costs = 0
        while exceeded:
            if(len(self.all_mappings) > 0):
                # checking for the departure time of the 
                if(self.all_mappings[0][0] <= step):
                    vnr_to_release = self.all_mappings[0]
                    resources.append(vnr_to_release[1]) # the first index is for departure time
                    self.all_mappings = self.all_mappings[1:]
                else:
                    exceeded = False
            else:
                exceeded = False

        for resource in resources:
            sub = copy.deepcopy(self.curr_state)
            for x in range(len(resource['sub_node_ind'])):
                sub['nodes'][resource['sub_node_ind'][x]]['cpu'] += resource['cpu_mem'][x][0]
                sub['nodes'][resource['sub_node_ind'][x]]['mem'] += resource['cpu_mem'][x][1]
        
            for x in range(len(resource['link_ind'])):
                for y in resource['link_ind'][x]:
                    sub['links'][y]['bw'] += resource['bw'][x]
            self.curr_state = sub
        
        for resource in resources:
            rev, cost = self.return_rev_cost(resource)
            revenues += rev
            costs += cost
        
        return revenues, costs
    
    def return_rev_cost(self, mapp):
        revenue = 0
        cost = 0
        for cpu_mem in mapp['cpu_mem']:
            revenue = cpu_mem[0] + cpu_mem[1]
        cost = copy.deepcopy(revenue)
        for u in range(len(mapp['paths'])):
            revenue += mapp['bw'][u]
            cost += (mapp['bw'][u] * (len(mapp['paths'][u]) - 1))
        
        return revenue, cost

    def return_avg_util(self):
        utils = []
        for node in self.curr_state['nodes']:
            util = 1 - (node['cpu'] / node['cpu_max'])
            utils.append(util)
        return np.mean(utils)

# from graph_gen import GraphGen
# from time_sim import Time_Sim
# import random

# # cpu_range, mem_range, bandwidth_range
# values_for_subs = [
#     [[50, 100], [64, 128], [50, 120]],
#     [[60, 120], [50, 100], [60, 100]],
#     [[80, 160], [64, 128], [50, 120]]
#     ]

# # number_of_nodes, cpu_req_range, mem_req_range, bandwidth_req_range
# values_for_vnrs = [
#     [2,10], [10,20], [10,20], [15,20]
# ]
# n_sub_graphs = 1

# gr_gen = GraphGen(sub_nodes=20, max_vnr_nodes=8, prob_of_link=0.1, sub_values=values_for_subs, vnr_values=values_for_vnrs)
# time_sim = Time_Sim()
# sub_graphs = gr_gen.make_cnst_sub_graphs(n_sub_graphs)
# resource_allocator = VNE_on_sub(sub_graphs[0])

# for x in range(10):
#     vnr = gr_gen.gen_rnd_vnr()
#     #pass the vnr to the low level agent
#     performed_actions = []
#     print(x)
#     # the resource allocator of the substrate that is chosen
#     resource_allocator.release_resources(x)
#     resource_allocator.receive_vnr(vnr=vnr)
#     resource_allocator.reset_map()    
#     for y in range(len(vnr['nodes'])):
#         # choose a low level action to embed the nodes of the vnr
#         action = None
#         while True:
#             if action in performed_actions or action is None:
#                 action = random.randint(0, 9)
#             else:
#                 break
#         performed_actions.append(action)
        
#         # provides also the graph after embedding this one VNR node
#         _, embedded = resource_allocator.embed_node(action, y)
    
#     # embedded true only if all the links are embedded and all the nodes are embedded
#     embedded = resource_allocator.embed_link()
#     # mapping is giving only if all the nodes and links are embedded otherwise an empty mapping is given
#     mapp = resource_allocator.get_mapping()
#     # print('vnr : ')
#     # print(vnr)
#     # print('mapp_for_vnr : ')
#     # print(mapp)
#     # only changed if all the nodes and links are embedded with the map that has been built till now
#     resource_allocator.change_sub()
#     # print('current_state_of_substrate : ')
#     # print(resource_allocator.curr_state)
#     # store the map with the given departure time
#     resource_allocator.store_map(x+5)
#     # print('all_mappings : ')
#     # print(resource_allocator.all_mappings)
#     # print('\n')

# # print(resource_allocator.curr_state)
# # print(resource_allocator.all_mappings)

    
