# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for graph neural network."""

import torch
import torch.nn as nn
import torch_scatter

from . import base_mp_gnn #, normalization

class GNN(nn.Module):
    """Model for GNN"""

    def __init__(self, learned_model, nfeat_in=2, nfeat_out=2, nfeat_out_global=0, args=None):
        super().__init__()
        self.nfeat_in=nfeat_in; self.nfeat_out=nfeat_out
        self.nfeat_out_global = nfeat_out_global
        self._learned_model = learned_model
        self.args = args

    def preprocess(self, x, is_training=False):
        """Builds input graph."""
        # construct graph nodes
        mesh_edges = base_mp_gnn.EdgeSet(
            name='mesh_edges',
            features=x.edge_features,#self._edge_normalizer(edge_features, is_training),
            receivers=x.edge_index[0],
            senders=x.edge_index[1])
        x.edge_sets=[mesh_edges]
        return x
        # return base_mp_gnn.MultiGraph(
        #     node_features=x.node_features,
        #     edge_sets=[mesh_edges])

    def forward(self, inputs, **kwx):
        graph = self.preprocess(inputs, is_training=False)
        node_output = self._learned_model(graph)
        if self.nfeat_out_global > 0:
            global_output = torch_scatter.scatter_mean(node_output[:, :self.nfeat_out_global], inputs.batch, dim=0)
            node_output = node_output[:, self.nfeat_out_global:]
        else:
            global_output = None
        return node_output#, global_output

