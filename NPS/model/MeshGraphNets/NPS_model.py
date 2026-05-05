# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for neural phase simulation."""

import torch
import torch.nn as nn

from . import common, base_mp_gnn #, normalization
from .common import output_by_method
from NPS.model.common import vector_pbc

class BaseNPSGNNModel(nn.Module):
    """Model for NPS simulation."""

    def __init__(self, learned_model, dim=2, periodic=False, nfeat_in=2, nfeat_out=2, unique_op=True, feat_out_method=None):
        super().__init__()
        self.dim=dim
        self.periodic=periodic
        self.nfeat_in=nfeat_in; self.nfeat_out=nfeat_out
        self.feat_out_method = feat_out_method
        self.unique_op = unique_op
        self._learned_model = learned_model
        self._output_normalizer = nn.Identity( #normalization.Normalizer(
            size=nfeat_out, name='output_normalizer')
        self._node_normalizer = nn.Identity( #normalization.Normalizer(
            size=nfeat_in+common.NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = nn.Identity( #normalization.Normalizer(
            size=dim+1, name='edge_normalizer')
        self.nfeat_node_embedding = common.NodeType.SIZE

    def preprocess(self, inputs, is_training=False):
        """Builds input graph."""
        # construct graph nodes
        if self.nfeat_node_embedding:
            node_type = nn.functional.one_hot(inputs['node_type'][:, 0], self.nfeat_node_embedding)
            node_features = torch.cat([inputs['x'], node_type], dim=-1)
        else:
            node_features = inputs['x']

        # construct graph edges
        if 'cells' in inputs:
            senders, receivers = common.triangles_to_edges(inputs['cells'], unique_op=self.unique_op)
        else:
            senders, receivers = inputs['edge_index'].t()
        relative_mesh_pos = inputs['mesh_pos'][senders] - inputs['mesh_pos'][receivers]
        # displacement vector under periodic boundary condition
        if self.periodic:
            relative_mesh_pos = vector_pbc(relative_mesh_pos, inputs['lattice'], inputs['inv_lattice'])
        edge_features = torch.cat([
            relative_mesh_pos,
            torch.linalg.norm(relative_mesh_pos, dim=-1, keepdim=True)], dim=-1)

        mesh_edges = base_mp_gnn.EdgeSet(
            name='mesh_edges',
            features=edge_features,#self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)
        return base_mp_gnn.MultiGraph(
            node_features=node_features,#self._node_normalizer(node_features, is_training),
            edge_sets=[mesh_edges])

    def forward(self, inputs, **kwx):
        graph = self.preprocess(inputs, is_training=False)
        per_node_network_output = self._learned_model(graph)
        return self._update(inputs, per_node_network_output)

    def loss(self, inputs):
        """L2 loss."""
        graph = self.preprocess(inputs, is_training=True)
        network_output = self._learned_model(graph)

        # build target x change
        cur_x = inputs['x']
        target_x = inputs['target|x']
        target_x_change = target_x - cur_x
        target_normalized = self._output_normalizer(target_x_change)

        # build loss
        node_type = inputs['node_type'][:, 0]
        loss_mask = torch.logical_or(node_type == common.NodeType.NORMAL,
                                    node_type == common.NodeType.OUTFLOW)
        error = torch.sum((target_normalized - network_output)**2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        # x_update = self._output_normalizer.inverse(per_node_network_output)
        x_update = per_node_network_output
        # integrate forward
        cur_x = inputs['xtime'][...,-1]
        # return cur_x + x_update
        return output_by_method(cur_x, x_update, method=self.feat_out_method)
