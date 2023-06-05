from itertools import combinations, groupby
import random
import networkx as nx
import pandas as pd
import json
import copy
import sys

class GraphGen:
    def __init__(self, n_subs=None, sub_nodes=None, max_vnr_nodes=6, prob_of_link=None, vnr_values=None) -> None:
        self.n_subs = n_subs
        self.sub_nodes = sub_nodes
        self.max_vnr_nodes = max_vnr_nodes
        self.prob_of_link = prob_of_link
        self.sub_values = [
        [16, 32, 48],
        [32, 64, 128],
        [256, 512, 1024]
        ]
        self.vnr_values = [
            [3, 6],
            [2, 4],
            [2,4,8],
            [32, 48]
        ]if vnr_values is None else vnr_values
    
    def gnp_random_connected_graph(self, n, p):
        #Generates a random undirected graph, similarly to an Erdős-Rényi 
        #graph, but enforcing that the resulting graph is conneted
        edges = combinations(range(n), 2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if p <= 0:
            return G
        if p >= 1:
            return nx.complete_graph(n, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge)
            for e in node_edges:
                if random.random() < p:
                    G.add_edge(*e)
        return G

    def make_cnst_sub_graphs(self):
        graphs_ = []
        df = pd.DataFrame(columns=['graphs'])
        for x in range(self.n_subs): # number of substrates we want
            gr = self.gnp_random_connected_graph(self.sub_nodes, self.prob_of_link)
            G = nx.node_link_data(gr)
            for node in G['nodes']:
                node['cpu_max'] = random.choice(self.sub_values[0])
                node['cpu'] = copy.copy(node['cpu_max'])
                node['mem_max'] = random.choice(self.sub_values[1])
                node['mem'] = copy.copy(node['mem_max'])
            for link in G['links']:    
                link['band_max'] = random.choice(self.sub_values[2])
                link['bw'] = copy.copy(link['band_max'])
            df.loc[x] = [json.dumps(G)]
            graphs_.append(G)
        
        # if this function is getting called from the main function
        df.to_csv('data/sub_graphs_original.csv', index=False)
        return graphs_, df
    
    def gen_rnd_vnr(self):
        gr = self.gnp_random_connected_graph(random.randint(*self.vnr_values[0]), self.prob_of_link)
        G = nx.node_link_data(gr)
        G['cpu_max'] = self.vnr_values[1][-1]
        G['mem_max'] = self.vnr_values[2][-1]
        G['band_max'] = self.vnr_values[3][-1]        
        for node in G['nodes']:
            node['cpu'] = random.choice(self.vnr_values[1])
            node['mem'] = random.choice(self.vnr_values[2])
        for link in G['links']:
            link['bw'] = random.choice(self.vnr_values[3])
        return G
    
    def load_sub_graphs(self, path):
        df = pd.read_csv( path + '/sub_graphs_used.csv')
        grs = []
        for x in range(self.n_subs):
            grs.append(json.loads(df.loc[x][0]))
        return grs


# cpu_range, mem_range, bandwidth_range
# values_for_subs = 

# number_of_nodes, cpu_req_range, mem_req_range, bandwidth_req_range
# values_for_vnrs = 

# gr_gen = GraphGen(n_subs=3, sub_nodes=20, prob_of_link=0.5)#, sub_values=values_for_subs, vnr_values=values_for_vnrs)
# gr_, _ = gr_gen.make_cnst_sub_graphs()
# print(gr_)
