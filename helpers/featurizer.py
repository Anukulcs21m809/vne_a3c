import numpy as np
import torch
from torch_geometric.data import Data
from helpers.graph_gen import GraphGen

#collects all the node features for the substrate and vnr graphs
class Featurizer:
    
    @staticmethod
    def featurize(G, sub=1, high=1, initial=True):
        all_node_features = []
        for node_idx in range(len(G['nodes'])):
            all_features = []
            if(sub):
                all_features = Featurizer.featurize_sub_node(G, node_idx)
            else:
                all_features = Featurizer.featurize_vnr_node(G, node_idx, high, initial)
            all_node_features.append(list(all_features))
        all_node_features = np.asarray(all_node_features)
        return torch.tensor(all_node_features, dtype=torch.float32)

    @staticmethod
    def featurize_sub_node(G, node_idx):
        cpu = G['nodes'][node_idx]['cpu'] 
        mem = G['nodes'][node_idx]['mem'] 
        cpu_util = 1 - (cpu / G['nodes'][node_idx]['cpu_max'])
        mem_util = 1 - (mem / G['nodes'][node_idx]['mem_max'])
        avg_rem_band = 0
        avg_util_band = 0
        i = 0
        for link in G['links']:
            if node_idx == link['target'] or node_idx == link['source']:
                avg_rem_band += link['bw']
                avg_util_band += (1 - (link['bw'] / link['band_max']))
                i += 1
        avg_rem_band = (avg_rem_band / i) if i > 1 else avg_rem_band
        avg_util_band = (avg_util_band / i) if i > 1 else avg_util_band

        return (cpu, mem, avg_rem_band, cpu_util, mem_util, avg_util_band)
    
    # all these max values are from the resource range that we have specified earlier for the vnr 
    @staticmethod
    def featurize_vnr_node(G, node_idx, high, initial):
        cpu = G['nodes'][node_idx]['cpu'] 
        mem = G['nodes'][node_idx]['mem']
        i = 0
        avg_band = 0
        for link in G['links']:
            if node_idx == link['target'] or node_idx == link['source']:
                avg_band += link['bw']
                i += 1
        avg_band = (avg_band / i) if i > 1 else avg_band

        if high:
            return (cpu, mem, avg_band)
        else:
            if initial:
            # two_flags => current_flag -> (0 - not_current, 1 - current) , embedded_flag -> (-1 - not_embedded, 1 - embedded)
                return (cpu, mem, avg_band, 1, -1)
            else:
                return (cpu, mem, avg_band, 0, -1)
    
    # expected to get G as a Data object
    @staticmethod
    def change_flags(G, node_idx, embedded, final=False):
        x = G.x
        x[node_idx][3] = 0
        x[node_idx][4] = 1 if embedded else -1
        if not final:
            x[node_idx + 1][3] = 1
        new_data = Data(x=x, edge_index=G.edge_index)
        return new_data

    @staticmethod
    def get_adj_info(G):
        edge_indices = []
        for edge in G['links']:
            i = edge['source']
            j = edge['target']
            edge_indices += [[i, j], [j, i]]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    
    @staticmethod
    def make_data_obj(G, sub=1, high=1, initial=True):
        node_info = Featurizer.featurize(G, sub=sub, high=high, initial=initial)
        edge_info = Featurizer.get_adj_info(G)
        data_ = Data(
                x = node_info,
                edge_index=edge_info
            )
        return data_

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

# gr = gr_gen.gen_rnd_vnr()
# print(gr)
# ftr = Featurizer.featurize(gr, sub=0, high=0)
# print(ftr)
