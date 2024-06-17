import torch
import torch.nn as nn
from model.AGCRNCell import AGCRNCell

"""

AGCRN: Attention-Based Graph Convolutional Recurrent Network

Purpose:

This model is designed for sequential prediction tasks on graph-structured data, 
combining graph convolutions, recurrent networks, and attention mechanisms.
It effectively captures both spatial dependencies between nodes and temporal dependencies within sequences.


Key Components:

AGCRNCell:
A specialized recurrent cell that integrates graph convolutions and gating mechanisms.


AVWDCRNN:
A multi-layer recurrent encoder that stacks multiple AGCRNCells.

Key operations:
Takes input features (x), initial hidden states (init_state), and node embeddings.
Processes input sequentially through each layer.
Returns processed outputs and the final hidden states of each layer.


AGCRN:
The overall model architecture.
Key components:
node_embeddings: Learnable embeddings for nodes, used in graph convolutions.
encoder: The AVWDCRNN encoder.
end_conv: A convolutional layer for final prediction.
Forward Pass:

Initialization:
Initializes hidden states for the encoder.

Encoding:
Passes input features through the encoder to capture spatial and temporal patterns.

Prediction:
Extracts the last hidden state from the encoder.
Applies a convolutional layer to generate predictions for multiple time steps.
Reshapes output to match target dimensions.

Key Features:
Graph Convolutions: Incorporates graph structure into the recurrent process for better spatial understanding.
Recurrent Layers: Captures temporal dependencies for sequence modeling.
Attention Mechanisms: (Implicitly within AVWGCN) Focuses on relevant nodes and relationships.
Multi-Layer Architecture: Enables learning of hierarchical representations.


Common Use Cases:
Traffic forecasting on road networks
User behavior prediction on social networks
Demand forecasting in supply chains
Anomaly detection in sensor networks

Strengths:
Effectively captures both spatial and temporal dependencies.
Adapts to different node characteristics through node-specific weights and biases.
Attends to important relationships for refined feature learning.


Weaknesses:
Can be computationally expensive for large graphs.
Requires careful hyperparameter tuning.
Interpretability can be challenging due to the complexity of graph convolutions and attention

"""


class AVWDCRNN(nn.Module):

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        self.dcrnn_cells = nn.ModuleList() 

        """"
            explanation of nn.ModuleList():

            Purpose:
            It's a container in PyTorch's nn module that holds a sequential list of modules.
            It's specifically designed to manage a collection of neural network modules within a larger model.

            Key Features:
            Sequential Access: You can access individual modules by their index, just like a regular Python list.
            Module Management: It automatically registers all modules within it as submodules of the parent module.
            Gradient Handling: It ensures that gradients are properly propagated through all modules during backpropagation.
        """
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))

        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
            

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim

        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings) # update hidden state
                inner_states.append(state) # Appends the updated hidden state to inner_states.

            output_hidden.append(state) # Appends the final hidden state of the layer to output_hidden
            current_inputs = torch.stack(inner_states, dim=1) # Stacks the hidden states from all time steps 
                                                              # within the layer into a tensor and assigns it to 
                                                              # current_inputs for the next layer.
            

        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden
    

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))

        return torch.stack(init_states, dim=0)   #(num_layers, B, N, hidden_dim)





"""
  AGCRN model pour la prediction
"""
class AGCRN(nn.Module):


    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph

        # Creates node embeddings to learn representations for nodes.
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        # Builds the AVWDCRNN encoder for spatial and temporal feature extraction.
        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers) 
       
        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)



    def forward(self, source, targets, teacher_forcing_ratio=0.5):  
        #source: B, T_1, N, D  (B: batch_size, )
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        # Extracts the last hidden state from the encoder's output, representing a summary of the encoded information:

        #CNN based predictor
        # Passes the last hidden state through the convolutional layer (end_conv) to generate predictions:
        output = self.end_conv((output))                         #B, T*C, N, 1

        # Reshapes the prediction tensor into the desired format
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output