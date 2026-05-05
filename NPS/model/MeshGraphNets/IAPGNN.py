# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for graph neural network."""

import torch
import torch.nn as nn
from torch_scatter import scatter
from .GNN_grad import GNN_grad


def register_args(parser):
    parser.add_argument('--n_species', type=int, default=1, help="number of species")
    parser.add_argument('--core', type=str, default="nequip_graphite", help="core GNN type")
    parser.add_argument('--precompute_bond_vec_correction', action="store_true", default=True)
    # parser.add_argument('--n_grad_in', type=str, default="1,4", help="dim of non-x and x (to differentiate WRT)")
    # parser.add_argument('--n_grad_gt', type=str, default="4,1,2", help="in GT: dim of grad, ignored sum, components to sum")
    # parser.add_argument('--UpDnSymm', type=int, default="0", help="Spin up/down symmetry")
    # parser.add_argument('--diag_only', type=int, default="0", help="off diagonal n_{s1,s2}")

def post_process_args(args):
    pass
    # args.n_grad_in = list(map(int, args.n_grad_in.split(",")))
    # args.n_grad_gt = list(map(int, args.n_grad_gt.split(",")))
    # args.n_tot_out = list(map(int, args.n_tot_out.split(",")))

def make_model(args):
    nhid = args.nfeat_hid
    if args.core == 'nequip_graphite':
        from graphite.nn.models.e3nn_nequip import NequIP
        from NPS.nn.nequip_embedding import InitialEmbedding
        emb = InitialEmbedding(num_species=args.n_species, pbc=args.periodic, cutoff=args.edge_cutoff,
                               precompute_bond_vec_correction=args.precompute_bond_vec_correction)
        learned_model = NequIP(
            init_embed     = emb,
            irreps_node_x  = '8x0e',
            irreps_node_z  = '8x0e',
            irreps_hidden  = f'{nhid}x0e + {nhid}x1e + {nhid//2}x2e',
            irreps_edge    = '1x0e + 1x1e + 1x2e',
            irreps_out     = '1x0e',
            num_convs      = args.n_mpassing,
            radial_neurons = [16, 64],
            num_neighbors  = 12,
        )
    elif args.core == 'MGN_IAP':
        from .MGN_IAP import MGN_IAP
        learned_model = MGN_IAP(
            input_size_node=args.n_species,
            input_size_edge=args.dim+1,
            output_size=1,
            latent_size_node=nhid, latent_size_edge=nhid,
            num_layers=2,
            message_passing_steps=args.n_mpassing,
            activation=args.act,
            # num_layers=args.nlayer_mlp,
            # nlayer_mlp_encdec=args.nlayer_mlp_encdec,
            # dropout=args.dropout,
            args=args)
    else:
        raise NotImplementedError(args.core)
    return IAPGNN(learned_model,
        nfeat_in=args.nfeat_in,
        nfeat_out=args.nfeat_out, nfeat_out_global=args.nfeat_out_global, args=args)


class IAPGNN(GNN_grad):
    def preprocess(self, inputs, *x, **kwx):
        g = inputs
        return g
        # g.edge_features = g.positions[g.edge_index[1]] - g.positions[g.edge_index[0]]

    def forward(self, inputs, target=None, criterion=nn.functional.mse_loss, **kwx):
        r = inputs.positions
        r.requires_grad_(True)
        graph = self.preprocess(inputs, is_training=False)
        with torch.enable_grad():
            node_output = self._learned_model(graph)
            enTot = scatter(node_output, graph.batch, dim=0)
            forces = -torch.autograd.grad(
                    [enTot.sum()],
                    r,
                    create_graph=True,#self.training,  # needed to allow gradients of this output during training
                )[0]
        if self.training:
            # print(f"{forces=}, {forces.shape=} \n{graph.forces=} {graph.forces.shape=}\n {graph.energy=} {graph.energy.shape=}");
            loss_grad =  criterion(forces, graph.forces)
            loss_tot  =  criterion(enTot, graph.energy)/graph.num_nodes
            return forces, [loss_grad, loss_tot]
        else:
            return forces

