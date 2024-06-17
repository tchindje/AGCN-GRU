import torch
import torch.nn.functional as F
import torch.nn as nn


"""
 explanation of the AVWGCN class and its code:


Purpose:

This class implements a graph convolutional network (GCN) layer with self-attention,
specifically an Attention-based Vertex-Weighted GCN (AVWGCN).
It's designed for tasks involving graph-structured data, where nodes have features and relationships.


Key Components:


__init__ Function:

Initializes parameters for the layer:
    cheb_k: Number of Chebyshev polynomials to approximate graph convolutions.
    weights_pool: Learnable weights for graph convolutions (parameterized by node embeddings).
    bias_pool: Learnable biases for graph convolutions (parameterized by node embeddings).

forward Function:
    Performs the graph convolution with attention:

Calculates attention scores:
    Uses node embeddings to compute pairwise similarities.
    Applies ReLU and softmax to obtain attention matrix (supports).
Generates Chebyshev polynomial approximations:
    Creates a set of matrices (support_set) representing different orders of graph connectivity.
Computes node-wise weights and biases:
    Uses node embeddings to dynamically generate weights and biases for each node.

Performs graph convolution:
    Applies Chebyshev filters (supports) to the input features (x).
    Combines filtered features with node-wise weights and biases.

Returns output:
    Provides updated node features (x_gconv) with incorporated graph structure and attention.

    
Key Features:

Self-attention: Captures long-range dependencies and important node relationships.
Vertex-weighted: Incorporates node-specific information for refined feature learning.
Chebyshev polynomial approximation: Efficiently handles graph convolutions.
Dynamic weights and biases: Adapts to different node characteristics.
Common Use Cases:

Node classification in graphs
Graph representation learning
Link prediction
Recommendation systems
Additional Notes:

dim_in specifies input feature dimension.
dim_out specifies output feature dimension.
embed_dim refers to dimension of node embeddings.
cheb_k controls the degree of polynomial approximation, typically set to 3.

"""


class AVWGCN(nn.Module):

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    
    def forward(self, x, node_embeddings):
        
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]

        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out

        return x_gconv