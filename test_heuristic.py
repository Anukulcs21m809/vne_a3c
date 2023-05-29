from helpers.graph_gen import GraphGen
import networkx as nx
from networkx import json_graph
import random
from helpers.network_embedder import VNE_on_sub
from helpers.time_sim import Time_Sim
from itertools import count
import numpy as np


   
class Heuristic:
    def __init__(self, arr_r, dep_t, res_size, test_=None) -> None:
        self.arr_r = arr_r
        self.dep_t = dep_t
        self.res_size = res_size
        self.test_ = test_
        self.g = GraphGen(n_subs=3, sub_nodes=20, max_vnr_nodes=10, prob_of_link=0.5, vnr_values=res_size)
        self.subs = self.g.load_sub_graphs('models/changed_sub_vnr_values')
        self.embedders = [VNE_on_sub(self.subs[x]) for x in range(len(self.subs))]
        self.sub_nx = [json_graph.node_link_graph(self.subs[x]) for x in range(len(self.subs))]
        self.steps = 1000
        self.time_sim = Time_Sim(mu=arr_r, dep_t=dep_t)
        self.neighbors = []
        self.count_of_nbrs = [[] for _ in range(len(self.subs))]
        self.count_of_nodes = [[0 for _ in range(len(self.subs[x]['nodes']))] for x in range(len(self.subs))]
        for i in range(len(self.subs)):
            neighbours = []
            for x in range(len(self.subs[i]['nodes'])):
                y = [n for n in self.sub_nx[i].neighbors(x)]
                neighbours.append(y)
                cnts = [0 for _ in range(len(y))]
                self.count_of_nbrs[i].append(cnts)
            self.neighbors.append(neighbours)

        self.revenues = [0,0,0]
        self.costs = [0,0,0]
        self.cum_rev = 0
    
    def rev_cost_ratio(self):
        tot_rev_ratio = np.mean([0 if c == 0 else r / c for r, c in zip(self.revenues, self.costs)])
        return tot_rev_ratio, self.cum_rev

    def run(self, heuristic='mine'):
        n_embedded = 0
        n_generated = 0
        i = 0
        self.time_sim.reset()
        arr_t = self.time_sim.ret_arr_time()

        ###################
        ac_ratio = []
        utils = []
        ###################

        for step in count():
            if step >= arr_t:
                n_generated += 1
                vnr = self.g.gen_rnd_vnr()
                # vnr_nx = json_graph.node_link_graph(vnr)
                if n_generated >= self.steps:
                    break
                self.embedder = self.embedders[i]
                if (i+1) % len(self.subs) == 0 and i != 0:
                    i = 0
                else:
                    i += 1

                # print(i)
                
                for k in range(len(self.embedders)):
                    revenues, costs = self.embedders[k].release_resources(step)
                    self.revenues[k] -= revenues
                    self.costs[k] -= costs
                    
                self.embedder.receive_vnr(vnr)
                self.embedder.reset_map()
                performed_actions = []
                for y in range(len(vnr['nodes'])):
                    action = None

                    if heuristic == 'mine':
                        while True:
                            if action in performed_actions or action is None:
                                if len(performed_actions) < 1:
                                    action = random.choices([x['id'] for x in self.subs[i]['nodes']], weights=[1/(len(self.subs[i]['nodes'])+ x) for x in self.count_of_nodes[i]])[0]
                                    self.count_of_nodes[i][action] += 1
                                else:
                                    action = random.choices(self.neighbors[i][performed_actions[-1]], weights=[1/(len(self.neighbors[i][performed_actions[-1]]) + x) for x in self.count_of_nbrs[i][performed_actions[-1]]])[0]
                                    ind = self.neighbors[i][performed_actions[-1]].index(action)
                                    self.count_of_nbrs[i][performed_actions[-1]][ind] += 1
                                    if len(performed_actions) == len(vnr['nodes'])-1:
                                        break
                            else:
                                break
                    else:
                        while True:
                            if action in performed_actions or action is None:
                                action = random.randint(0, len(self.subs[i]['nodes'])-1)
                            else:
                                break
                        performed_actions.append(action)

                    _, embedded = self.embedder.embed_node(action, y)
                embedded = self.embedder.embed_link()
                mapp, fully_embedded = self.embedder.get_mapping()

                rev, cost = self.embedder.return_rev_cost(mapp)
                # print(rev, cost)
                self.revenues[i] += rev
                self.costs[i] += cost
                self.cum_rev += rev

                self.embedder.change_sub()
                dep_t = self.time_sim.ret_dep_time(arr_t)
                self.embedder.store_map(dep_t)
                self.embedders[i] = self.embedder

                arr_t = self.time_sim.ret_arr_time()

                if fully_embedded:
                    n_embedded += 1
            
                utils.append(np.mean([self.embedders[x].return_avg_util() for x in range(len(self.embedders))]))
                ac_ratio.append(n_embedded / n_generated)

        rev_cost_, cum_rev_ = self.rev_cost_ratio()
        # print(self.revenues, self.costs)

        f = open('result_text_files/results_heuristic.txt', 'a')
        f.write('test:{}'.format(self.test_) + '\n')
        f.write(str('algo:{}'.format('heuristic')) + '\n')
        f.write(str(self.arr_r) + '\n')
        f.write(str(self.dep_t) + '\n')
        f.write(str(self.res_size) + '\n')
        f.write(str('accept_ratio:{}'.format(str(ac_ratio[-1]))) + '\n')
        f.write(str('rev_cost:{}'.format(str(np.mean(rev_cost_)))) + '\n')
        f.write(str('cum_rev:{}'.format(str(np.mean(cum_rev_)))) + '\n\n')
        f.close()

        f = open('result_text_files/util_results_heuristic.txt', 'a')
        f.write('test:{}'.format(self.test_)  + '\n')
        f.write(str('algo:{}'.format('heuristic')) + '\n')
        f.write(str(self.arr_r) + '\n')
        f.write(str(self.dep_t) + '\n')
        f.write(str(self.res_size) + '\n')
        f.write('util : {}'.format(utils))
        f.write('\n\n')
        f.close()

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
        sub_embedd_values = [0, 0 , 0]
        if i == 0:
            arr_r = val
            test = 'arr_rate'
        elif i == 1:
            dep_t = val
            test = 'dept_time'
        else:
            res_size = val
            test = 'vnr_size'
        h = Heuristic(arr_r, dep_t, res_size, test_=test)
        h.run(heuristic='mine')
    i += 1



    
                                               
                
            


