import torch
from neural_networks.layers import AttentionLayer, NeuralTensorLayer, GCNNBlock
from torch.distributions import Categorical
import random
import math

class LowNetwork(torch.nn.Module):
    def __init__(self, args) -> None:
        super(LowNetwork, self).__init__()
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
                action = random.randint(0, self.args['n_sub_nodes']-1)
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
        # print('policy : ' , pi)
        # print('values : ', values)

        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        dist = Categorical(pi)
        log_probs = dist.log_prob(actions)
        # print(log_probs)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss
    
    ########################
    
    def setup_layers(self):
        sub_ftrs = self.args['sub_ftrs']
        vnr_ftrs = self.args['vnr_ftrs_low']
        self.gcn_sub = GCNNBlock(sub_ftrs, self.args)
        self.gcn_vnr = GCNNBlock(vnr_ftrs, self.args)
        self.attention = AttentionLayer(self.args)
        self.ntn_layer = NeuralTensorLayer(self.args)
        self.shared_fc_layer_1 = torch.nn.Linear(self.args['ntn_neurons'], self.args['shared_layer_1_n'])
        self.policy_head = torch.nn.Linear(self.args['shared_layer_1_n'], self.args['n_sub_nodes'])
        self.softmax = torch.nn.Softmax(dim=1)
        self.value_head = torch.nn.Linear(self.args['shared_layer_1_n'], 1)
    
    # the expected format for the state -> [sub_data_obj, vnr_data_obj_with_flags]
    def forward(self, state):
        abs_sub_ftrs = self.gcn_sub(state[0])
        abs_vnr_ftrs = self.gcn_vnr(state[1])
        pooled_sub_ftrs = self.attention(abs_sub_ftrs)
        pooled_vnr_ftrs = self.attention(abs_vnr_ftrs)
        scores = self.ntn_layer(pooled_sub_ftrs, pooled_vnr_ftrs)
        scores = torch.t(scores)

        final_scores = torch.nn.functional.relu(self.shared_fc_layer_1(scores))
        final_scores = torch.nn.functional.dropout(final_scores, p=self.args['dropout'], training=self.training)
        action_dist = self.policy_head(final_scores)
        action_dist = self.softmax(action_dist)
        value = self.value_head(final_scores) 

        return action_dist, value
    
    