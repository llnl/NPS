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
from NPS.model.common import vector_pbc


def register_args(parser):
    parser.add_argument('--nfeat_onehot', type=int, default=None, help='onehot embedding dim for node type')
    parser.add_argument('--nfeat_edge_in', type=int, default=0, help='Note: input edge feature should be: args.dim+1+args.nfeat_edge_in')
    parser.add_argument('--nlayer_mlp_encdec', type=int, default=2, help='MLP layer in GNN encoder/decoder')

def post_process_args(args):
    assert not args.channel_first, ValueError(f'GNN demands channel last')
    assert not args.RNN, ValueError('MGN is feedforward only')

def make_model(args):
    core_model_type = base_mp_gnn.EncodeProcessDecode
    learned_model = core_model_type(
            input_size_node=2+3, input_size_edge=args.dim+1+args.nfeat_edge_in,
            output_size=3,
            activation=args.act,
            latent_size_node=args.nfeat_hid, latent_size_edge=args.nfeat_hid_edge,
            num_layers=args.nlayer_mlp,
            nlayer_mlp_encdec=args.nlayer_mlp_encdec,
            dropout=args.dropout,
            message_passing_steps=args.n_mpassing)
    return GNN_mobility_ds5(learned_model, 
            nfeat_in=args.nfeat_in,
            nfeat_out=args.nfeat_out, nfeat_out_global=0)
 

class GNN_mobility_ds5(GNN):
    def preprocess_dataset(self, dataset, args):
        from torch_geometric.data import Data
        for i, d in enumerate(dataset):
            # print(d)
            edge_index = d['edge_index']
            nodes = d['pos']
            edge_index = torch.cat((edge_index, torch.flip(edge_index,[0])), 1)
            edge_vec = vector_pbc(nodes[edge_index[1]] - nodes[edge_index[0]], d['cell_len']).float()/1e3
            burgers = torch.from_numpy(d['burgers'])
            burgers = torch.cat((burgers, -burgers)).float()
            seg_norm = torch.from_numpy(d['seg_normal'])
            seg_norm = torch.cat((seg_norm, seg_norm)).float() # note EVEN parity
            edge_features = torch.cat([
                burgers,  torch.linalg.norm(burgers, dim=-1, keepdim=True),
                seg_norm, torch.linalg.norm(seg_norm, dim=-1, keepdim=True),
                edge_vec, torch.linalg.norm(edge_vec, dim=-1, keepdim=True)], dim=-1)
            force = d['force']/1e10
            node_features = torch.cat((nn.functional.one_hot(d['node_type'], args.nfeat_onehot).float(), force), -1)
            velocity = d['velocity']/1e10
            # dataset[i] = {'edge_index':edge_index, 'edge_features':edge_features, 'node_features':node_features, 'node_y':velocity}
            dataset[i] = Data(edge_index=edge_index, edge_features=edge_features, node_features=node_features, node_y=velocity)
            # print([(k, v.shape) for k,v in dataset[i].items()])
        if False:
            import numpy as np
            for k in dataset[0]:
                if k == 'edge_index': continue
                np.save(f'{k}.npy', torch.cat([d[f'{k}'] for d in dataset]))
        return dataset

    """Output a mobility"""
    def forward(self, inputs, **kwx):
        graph = self.preprocess(inputs, is_training=False)
        # mobility = self._learned_model(graph)
        # mobility = rearrange(mobility, 'b (d a) -> b d a', d=self.nfeat_out)
        # node_output = torch.bmm(mobility, inputs['force'][...,None])[...,0]
        # if self.nfeat_out_global > 0:
        #     raise "NOT implemented"
        # else:
        #     global_output = None
        node_output = self._learned_model(graph)
        return node_output, None

