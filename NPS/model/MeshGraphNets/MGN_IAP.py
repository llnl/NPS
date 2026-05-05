# Lint as: python3
# pytorch port 
# ============================================================================
"""Core learned graph net model."""

# import collections
# import functools
import torch
import torch.nn as nn
# from torch_scatter import scatter
# from NPS.model.common import MLP_


from GNN.MeshGraphNets.base_mp_gnn import EncodeProcessDecode, MultiGraph, EdgeSet

class MGN_IAP(EncodeProcessDecode):
    def init_encoder_node(self):
        self.encoder_node = nn.Embedding(self.args.n_species, self.args.nfeat_hid)

    def encode(self, data):
        """Encodes node and edge features into latent features."""
        node_latents = self.encoder_node(data.x)
        edge_attr = data.positions[data.edge_index[1]] - data.positions[data.edge_index[0]]
        if self.args.periodic:
            if self.args.precompute_bond_vec_correction:
                edge_attr = edge_attr + data.edge_vec_correction
            else:
                edge_attr -= torch.bmm(torch.round(torch.bmm(edge_attr.detach()[:,None], data.inv_lattice[data.edge_index[0]])), data.lattice[data.edge_index[0]])[:,0]
        edge_attr = torch.cat((edge_attr, edge_attr.norm(dim=-1,keepdim=True)), -1)
        new_edges_sets = []
        for i in range(1):
            latent = self.encoders_edge[i](edge_attr)
            senders, receivers = data.edge_index
            new_edges_sets.append(EdgeSet(features=latent, name='mesh_edges',
                                          receivers=receivers, senders=senders))
        return MultiGraph(node_latents, new_edges_sets)
