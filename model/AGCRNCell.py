import torch
import torch.nn as nn
from model.AGCN import AVWGCN

"""

AGCRNCell: Combining Graph Convolutions and Recurrent Neural Networks

Purpose:

This class implements a specialized recurrent cell for handling graph-structured data, integrating graph convolutions into a recurrent architecture.
It's designed to capture both temporal dependencies within sequences and structural relationships between nodes in a graph.
Key Components:

__init__ Function:

Initializes parameters for the cell:
node_num: Number of nodes in the graph.
dim_in: Input feature dimension for each node.
dim_out: Hidden state dimension for each node.
cheb_k: Number of Chebyshev polynomials for graph convolutions.
embed_dim: Dimension of node embeddings.
Creates two AVWGCN layers (gate and update) for graph convolutions.
forward Function:

Performs a single step of the recurrent cell:
Combines input and previous state: Concatenates current input features (x) with the previous hidden state (state).
Applies gate:
Passes the combined input through the gate AVWGCN layer.
Splits the output into update gate (z) and reset gate (r) using a sigmoid activation.
Calculates candidate state:
Multiplies the previous state (state) by the update gate (z).
Concatenates the result with the input features (x).
Passes through the update AVWGCN layer and applies a tanh activation.
Combines previous state and candidate:
Uses the reset gate (r) to control the combination of the previous state (state) and the candidate state (hc).
Returns updated hidden state: Outputs the new hidden state (h) for the current time step.
init_hidden_state Function:

Initializes the hidden state tensor with zeros for a given batch size.
Key Features:

Graph Convolutional Layers: Employs AVWGCN layers to incorporate graph structure and node relationships into the recurrent process.
Gate Mechanisms: Uses update and reset gates to control information flow and prevent gradient issues, similar to LSTM or GRU cells.
Node-Specific Weights and Biases: AVWGCN layers adapt to different node characteristics through dynamic weights and biases.
Common Use Cases:

Traffic forecasting on road networks
User behavior prediction on social networks
Molecular property prediction in chemistry
Anomaly detection in sensor networks
Diagram:

[Insert a diagram illustrating the AGCRNCell's components and data flow]



"""

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)