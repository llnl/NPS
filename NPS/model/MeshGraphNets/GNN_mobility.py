# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for graph neural network."""

import torch
import torch.nn as nn
import torch_scatter

from MeshGraphNets import common, base_mp_gnn #, normalization
from MeshGraphNets.GNN import GNN
from einops import rearrange

class GNN_mobility(GNN):
    """Output a mobility"""
    def forward(self, inputs, **kwx):
        graph = self.preprocess(inputs, is_training=False)
        mobility = self._learned_model(graph)
        mobility = rearrange(mobility, 'b (d a) -> b d a', d=self.nfeat_out)
        node_output = torch.bmm(mobility, inputs['force'][...,None])[...,0]
        if self.nfeat_out_global > 0:
            raise "NOT implemented"
        else:
            global_output = None
        return node_output, global_output

