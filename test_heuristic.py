from helpers.graph_gen import GraphGen
import networkx as nx
from networkx import json_graph
import random
from helpers.network_embedder import VNE_on_sub
from helpers.time_sim import Time_Sim
from itertools import count

g = GraphGen(n_subs=1, sub_nodes=20, max_vnr_nodes=3, prob_of_link=0.5)
   
class Heuristic:
    def __init__(self) -> None:
        self.sub = g.make_cnst_sub_graphs()[0][0]
        self.embedder = VNE_on_sub(self.sub)
        self.sub_nx = json_graph.node_link_graph(self.sub)
        self.steps = 10000
        self.time_sim = Time_Sim(mu=0.04)
        self.neighbors = []
        self.count_of_nbrs = []
        self.count_of_nodes = [0 for _ in range(len(self.sub['nodes']))]
        for x in range(len(self.sub['nodes'])):
            y = [n for n in self.sub_nx.neighbors(x)]
            self.neighbors.append(y)
            cnts = [0 for _ in range(len(y))]
            self.count_of_nbrs.append(cnts)
    
    def run(self, heuristic='mine'):
        n_embedded = 0
        n_generated = 0
        self.time_sim.reset()
        arr_t = self.time_sim.ret_arr_time()
        for step in count():
            if step >= arr_t:
                n_generated += 1
                vnr = g.gen_rnd_vnr()
                # vnr_nx = json_graph.node_link_graph(vnr)
                if n_generated >= self.steps:
                    break

                self.embedder.release_resources(step)
                self.embedder.receive_vnr(vnr)
                self.embedder.reset_map()
                performed_actions = []
                for y in range(len(vnr['nodes'])):
                    action = None

                    if heuristic == 'mine':
                        while True:
                            if action in performed_actions or action is None:
                                if len(performed_actions) < 1:
                                    action = random.choices([x['id'] for x in self.sub['nodes']], weights=[1/(len(self.sub['nodes'])+ x) for x in self.count_of_nodes])[0]
                                    self.count_of_nodes[action] += 1
                                else:
                                    action = random.choices(self.neighbors[performed_actions[-1]], weights=[1/(len(self.neighbors[performed_actions[-1]]) + x) for x in self.count_of_nbrs[performed_actions[-1]]])[0]
                                    ind = self.neighbors[performed_actions[-1]].index(action)
                                    self.count_of_nbrs[performed_actions[-1]][ind] += 1
                                    if len(performed_actions) == len(vnr['nodes'])-1:
                                        break
                            else:
                                break
                    else:
                        while True:
                            if action in performed_actions or action is None:
                                action = random.randint(0, len(self.sub['nodes'])-1)
                            else:
                                break
                        performed_actions.append(action)

                    _, embedded = self.embedder.embed_node(action, y)
                embedded = self.embedder.embed_link()
                mapp = self.embedder.get_mapping()
                self.embedder.change_sub()
                dep_t = self.time_sim.ret_dep_time(arr_t)
                self.embedder.store_map(dep_t)

                arr_t = self.time_sim.ret_arr_time()

                if embedded:
                    n_embedded += 1

        print(self.embedder.curr_state)
        print(self.embedder.all_mappings)

        return n_embedded / n_generated

h = Heuristic()
print(h.run(heuristic='mine'))


    
                                               
                
            


