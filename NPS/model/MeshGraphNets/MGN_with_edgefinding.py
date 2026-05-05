import torch
import torch.nn as nn
from NPS.model.common import MLP_
from MeshGraphNets import base_mp_gnn, common
from MeshGraphNets.base_mp_gnn import MultiGraph


class EncodeProcessDecode_edgefinding(base_mp_gnn.EncodeProcessDecode):
    def __init__(self, *xargs, nfeat_onehot=2, cutoff=4.1, fixedbond=0, **kwargs):
        super().__init__(*xargs, **kwargs)
        self.nfeat_onehot = nfeat_onehot
        self.cutoff = cutoff
        self.fixedbond = fixedbond

    def preprocess(self, g, reuse_edge=False):
        """Builds input graph"""
        # Constructing MultiGraph for MGN

        node_features = nn.functional.one_hot(g.node_type, self.nfeat_onehot).float()
        # node_features = torch.cat([dx, node_type], dim=-1)

        # construct graph edges
        if reuse_edge and hasattr(g, 'edge_vec'):
            edge_index, edge_vec = g.edge_index, g.edge_vec
        else:
            edge_index, edge_vec = common.get_neighbor_list(g.pos, [100.]*3, False, self.cutoff, batch=g.ptr, bruteforce=True)
        # print(f'debug fixed bond {g["fixedbond_index"]}', g, g["fixedbond_type"], g["fixedbond_type"].__class__, g["fixedbond_type"][0].__class__)
        if self.fixedbond:
            # print(f'debug fixed bond {g["fixedbond_index"]}')
            edge_index, edge_type = common.add_edge(edge_index, g["fixedbond_index"], g["fixedbond_type"], g.pos.size(0))
            edge_type = nn.functional.one_hot(edge_type.long())
            edge_vec = g.pos[edge_index[1]] - g.pos[edge_index[0]]
        else:
            edge_type = edge_vec.new_zeros((edge_vec.size(0), 0))
        # relative_mesh_pos = inputs['mesh_pos'][senders] - inputs['mesh_pos'][receivers]
        # displacement vector under periodic boundary condition
        # if self.periodic:
        #     relative_mesh_pos = vector_pbc(relative_mesh_pos, inputs['lattice'], inputs['inv_lattice'])
        edge_features = torch.cat([
            edge_vec,
            torch.linalg.norm(edge_vec, dim=-1, keepdim=True),
            edge_type,
            ], dim=-1)

        mesh_edges = base_mp_gnn.EdgeSet(
            name='mesh_edges',
            features=edge_features,#self._edge_normalizer(edge_features, is_training),
            receivers=edge_index[0],
            senders=edge_index[1])
        return base_mp_gnn.MultiGraph(
            node_features=node_features,#self._node_normalizer(node_features, is_training),
            edge_sets=[mesh_edges])

    def forward(self, g_in, reuse_edge=False):
        """Encodes and processes a multigraph, and returns node features."""
        gammas = gammas.view(-1, 1)
        graph = self.preprocess(g_in, reuse_edge=reuse_edge)
        return super().forward(graph)

