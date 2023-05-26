########################
# This implementation of SimGNN has been taken from the 
# implementation by benedekrozemberczki 
# https://github.com/benedekrozemberczki
########################

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class AttentionLayer(torch.nn.Module):
    def __init__(self, args) -> None:
        super(AttentionLayer, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()
    
    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args['filters_3'], 
                                                             self.args['filters_3']))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
    
    # only takes float values
    def forward(self, gcn_embedding):
        global_context = torch.mean(torch.mm(gcn_embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(gcn_embedding, transformed_global.view(-1,1))) # shows the matrix as a 2D vector
        representation = torch.mm(torch.t(gcn_embedding), sigmoid_scores)
        return representation

class NeuralTensorLayer(torch.nn.Module):
    def __init__(self, args) -> None:
        super(NeuralTensorLayer, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args['filters_3'], self.args['filters_3'], self.args['ntn_neurons']))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args['ntn_neurons'], 2 * self.args['filters_3']))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args['ntn_neurons'], 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)
    
    def forward(self, gcn_embedding_1, gcn_embedding_2):
        # print(self.weight_matrix.shape)
        # print(self.weight_matrix_block.shape)
        # print(self.bias.shape)
        scoring = torch.mm(torch.t(gcn_embedding_1), self.weight_matrix.view(self.args['filters_3'], -1))
        scoring = scoring.view(self.args['filters_3'], self.args['ntn_neurons'])
        scoring = torch.mm(torch.t(scoring), gcn_embedding_2)
        combined_repr = torch.cat((gcn_embedding_1, gcn_embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_repr)
        scores = torch.relu(scoring + block_scoring + self.bias)
        return scores

class GCNNBlock(torch.nn.Module):
    def __init__(self, in_features, args) -> None:
        super(GCNNBlock, self).__init__()
        self.in_features = in_features
        self.args = args
        self.setup_layers()
    
    def setup_layers(self):
        self.gcnn_layer_1 = GCNConv(self.in_features, self.args['filters_1'])
        self.gcnn_layer_2 = GCNConv(self.args['filters_1'], self.args['filters_2'])
        self.gcnn_layer_3 = GCNConv(self.args['filters_2'], self.args['filters_3'])
    
    def forward(self, input):
        conv_1 = torch.nn.functional.relu(self.gcnn_layer_1(input.x, input.edge_index))
        conv_1 = torch.nn.functional.dropout(conv_1, p=self.args['dropout'], training=self.training)
        conv_2 = torch.nn.functional.relu(self.gcnn_layer_2(conv_1, input.edge_index))
        conv_2 = torch.nn.functional.dropout(conv_2, p=self.args['dropout'], training=self.training)
        conv_3 = self.gcnn_layer_3(conv_2, input.edge_index)
        return conv_3

# gcnn_block = GCNNBlock(3, {'filters_1': 8, 'filters_2': 4, 'filters_3': 2, 'dropout':0.5})
# x = torch.tensor([[1.2,3.4,4.5], [2.7,6.3,4.8]], dtype=torch.float32)
# edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
# data_ = Data(x=x, edge_index=edge_index)
# print(gcnn_block(data_))

# att = AttentionLayer({'filters_3':3})
# rand_ = torch.tensor(([[1.3,2.5,3.6], [3.3,4.4,5.6], [5.7,6.3,7.6]])) #inputs should be float values
# print(att(rand_))

# ntn = NeuralTensorLayer({'filters_3': 3, 'ntn_neurons':16})
# rand_1 = torch.tensor([1.3,2.5,3.6]).view(-1, 1)
# rand_2 = torch.tensor([3.4,5.6,7.8]).view(-1, 1)
# print(ntn(rand_1, rand_2))