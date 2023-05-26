import torch
from torch_geometric.data import Data
from neural_networks.layers import AttentionLayer, NeuralTensorLayer, GCNNBlock
from torch.distributions import Categorical
import random
import math

class HighNetwork(torch.nn.Module):
    def __init__(self, args) -> None:
        super(HighNetwork, self).__init__()
        self.args = args
        self.setup_layers()
        self.clear_memory()
        self.steps_done = 0
    
    #########################
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def remember(self, state, action , reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # return the index of the action with the highest prob
    def select_action(self, obs, train=True):
        pi, _ = self.forward(obs)
        if train:
            sample = random.random()
            eps_threshold = self.args['eps_end'] + (self.args['eps_start'] - self.args['eps_end']) * \
            math.exp(-1. * self.steps_done / self.args['eps_decay'])
            self.steps_done += 1
            if sample > eps_threshold:
                action = action = pi.argmax().item()
            else:
                action = random.randint(0, self.args['n_subs']-1)
            # dist = Categorical(pi)
            # action = dist.sample().numpy()[0]
        else:
            action = action = pi.argmax().item()
        return action

    def calc_R(self, done):
        states_batch = self.states
        pi = []
        v = []
        for x in range(len(states_batch)):
            pi_, v_ = self.forward(states_batch[x])
            pi.append(pi_)
            v.append(v_)
        
        pi = torch.cat(pi)
        v = torch.cat(v)
        
        R = v[-1]*(1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.args['gamma'] * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def calc_loss(self, done):
        actions = torch.tensor(self.actions, dtype=torch.float)
        returns = self.calc_R(done)
        states_batch = self.states
        pi = []
        values = []
        for x in range(len(states_batch)):
            pi_, v_ = self.forward(states_batch[x])
            pi.append(pi_)
            values.append(v_)
        
        pi = torch.cat(pi)
        values = torch.cat(values)

        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        dist = Categorical(pi)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss
       
    ########################
    
    def calc_feature_cnt(self):
        if self.args['histogram'] == True:
            self.feature_count = self.args['ntn_neurons'] + self.args['bins']
        else:
            self.feature_count = self.args['ntn_neurons']

    def setup_layers(self):
        self.n_subs = self.args['n_subs']
        sub_ftrs = self.args['sub_ftrs']
        vnr_ftrs = self.args['vnr_ftrs_high']
    
        self.calc_feature_cnt()
        # print(self.feature_count)
        # self.sub_gcnns = torch.nn.ModuleList([GCNNBlock(sub_ftrs, self.args) for _ in range(n_subs)])
        self.sub_gcnn = GCNNBlock(sub_ftrs, self.args)
        self.vnr_gcnn = GCNNBlock(vnr_ftrs, self.args)
        # self.sub_attentions = torch.nn.ModuleList([AttentionLayer(self.args) for _ in range(n_subs)])
        self.attention = AttentionLayer(self.args)
        # self.subs_ntn = torch.nn.ModuleList([NeuralTensorLayer(self.args) for _ in range(n_subs)])
        self.ntn_layer = NeuralTensorLayer(self.args)
        self.shared_fc_layer1 = torch.nn.Linear(self.feature_count * self.n_subs, self.args['shared_layer_1_n']) # neurons for the shared layer
        self.shared_fc_layer2 = torch.nn.Linear(self.args['shared_layer_1_n'], self.args['shared_layer_2_n'])
        self.policy_head = torch.nn.Linear(self.args['shared_layer_2_n'], self.n_subs)
        self.softmax = torch.nn.Softmax(dim=1)
        self.value_head = torch.nn.Linear(self.args['shared_layer_2_n'], 1) 
    
    def calculate_histogram(self, abstract_f_1, abstract_f_2):
        scores = torch.mm(abstract_f_1, abstract_f_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args['bins'])
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    # the expected format for the state -> [sub_data_obj_1, sub_data_obj_2, sub_data_obj_3, vnr_data_obj]
    def forward(self, state):
        abs_features_sub = [self.sub_gcnn(state[i]) for i in range(self.n_subs)]
        abs_features_vnr = self.vnr_gcnn(state[-1])

        if self.args['histogram'] == True:
            histograms = [self.calculate_histogram(abs_features_sub[i] , torch.t(abs_features_vnr)) for i in range(self.n_subs)]

        pooled_sub_ftrs = [self.attention(abs_features_sub[i]) for i in range(self.n_subs)]
        pooled_vnr_ftrs = self.attention(abs_features_vnr)
        scores = [self.ntn_layer(pooled_sub_ftrs[i], pooled_vnr_ftrs) for i in range(self.n_subs)]
        scores = [torch.t(scores[i]) for i in range(self.n_subs)]
        
        if self.args['histogram'] == True:
            scores = [torch.cat((scores[i], histograms[i]), dim=1).view(1, -1) for i in range(self.n_subs)]
        final_scores = scores[0]
        for x in range(1, self.n_subs):
            final_scores = torch.cat((final_scores, scores[x]), dim=1).view(1, -1)
        
        final_scores = torch.nn.functional.relu(self.shared_fc_layer1(final_scores))
        final_scores = torch.nn.functional.dropout(final_scores, p=self.args['dropout'], training=self.training)
        final_scores = torch.nn.functional.relu(self.shared_fc_layer2(final_scores))
        final_scores = torch.nn.functional.dropout(final_scores, p=self.args['dropout'], training=self.training)
        action_dist = self.policy_head(final_scores)
        action_dist = self.softmax(action_dist)
        value = self.value_head(final_scores)

        return action_dist, value
    


# args_ = {'filters_1': 8,
#           'filters_2': 4, 
#           'filters_3': 2, 
#           'dropout':0.5,
#           'ntn_neurons':16,
#           'bins':8,
#           'histogram':True,
#           'n_subs':3,
#           'sub_ftrs':6,
#           'vnr_ftrs_high':3,
#           'vnr_ftrs_low':5,
#           'shared_layer_1_n':32,
#           'shared_layer_2_n':16,
#           'n_sub_nodes':30}

# sub_1 = torch.tensor([[2,3,5,0.9,0.8,0.7],
#          [5,4,3,0.3,0.5,0.7],
#          [3,2,4,0.5,0.4,0.9],
#          [4,5,4,0.3,0.4,0.2]], dtype=torch.float32)
# sub_2 = torch.tensor([[5,4,3,0.1,0.4,0.5],
#           [4,3,3,0.4,0.5,0.2],
#           [4,2,3.5,0.7,0.5,0.8],
#           [2,2,2,0.6,0.4,0.8]], dtype=torch.float32)
# sub_3 = torch.tensor([[4,3,5,0.6,0.8,0.5],
#           [2,2,5.5,0.3,0.4,0.6],
#           [2,5,4,0.4,0.5,0.8],
#           [3,4,2,0.3,0.6,0.6]], dtype=torch.float32)
# edge_1 = torch.tensor([[0,2,1,2,2,3],
#                       [2,0,2,1,3,2]], dtype=torch.long)
# edge_2 = torch.tensor([[0,2,1,2,1,3],
#                        [2,0,2,1,3,1]], dtype=torch.long)
# edge_3 = torch.tensor([[0,1,1,2,1,3],
#                        [1,0,2,1,3,1]], dtype=torch.long)
# vnr = torch.tensor([[1,2,2],
#                    [2,1,2]], dtype=torch.float32)
# vnr_l = torch.tensor([[1,2,2,1,-1],
#                       [2,1,2,0,-1]], dtype=torch.float32)
# edge_vnr = torch.tensor([[0,1],
#                          [1,0]], dtype=torch.long)

# data_1 = Data(x=sub_1, edge_index=edge_1)
# data_2 = Data(x=sub_2, edge_index=edge_2)
# data_3 = Data(x=sub_3, edge_index=edge_3)
# data_vnr = Data(x=vnr, edge_index=edge_vnr)
# data_vnr_l = Data(x=vnr_l, edge_index=edge_vnr)
# all_data_hn = [data_1, data_2, data_3, data_vnr]
# all_data_ln = [data_1, data_vnr_l]

# hn = HighNetwork(args_)
# print(hn(all_data_hn))
# ln = LowNetwork(args_)
# print(ln(all_data_ln))


        

