# Lint as: python3
# pytorch port 
# ============================================================================
"""Core learned graph net model."""

# import collections
import functools
import numpy as np
import torch
import torch.nn as nn
# from torch_scatter import scatter
# from NPS.model.common import MLP_

from torchdeq import get_deq, add_deq_args
from torchdeq.norm import apply_norm, reset_norm

# from deq import get_deq
# from deq.norm import apply_weight_norm, reset_weight_norm
# from deq.layer_utils import DEQWrapper
# from deq.arg_utils import add_deq_args

def register_args(parser):
    # from NPS.model.deq_toy import register_args as _r_a
    # _r_a(parser)
    # add_deq_args(parser)
    parser.add_argument('--deq_args', type=str, default="")
    parser.add_argument('--nlayer_enc_nondeq', type=int, default=0, help="add normal (non-weight-shared) encoding layers")
    # parser.add_argument('--n_deq_pretrain', default='', help="DEQ in pre-training mode")
    # print(f"debug  registered **** \n **** {dir(parser)=}")

def post_process_args(args):
    # from NPS.model.deq_toy import post_process_args as _pp_a
    # _pp_a(args)
    # args.n_deq_pretrain = list(map(int, filter(None, args.n_deq_pretrain.split(','))))[:2]
    # if len(args.n_deq_pretrain)>1: args.n_deq_pretrain[1] += 1
    import argparse, shlex
    # print(f"{args=}")
    parser_dummy = argparse.ArgumentParser()
    # print(f"debug post process")
    add_deq_args(parser_dummy)
    args.deq_args = parser_dummy.parse_args(shlex.split(args.deq_args))
    # print(f"{parser_dummy=}")
    # print(f"debug post process deq")
    # deq_keys = tuple(vars(parser_dummy.parse_args(shlex.split(args.deq_args))).keys())
    # args.deq_args = {k:args[k] for k in deq_keys}

from GNN.MeshGraphNets.base_mp_gnn import GraphNetBlock, EncodeProcessDecode, EdgeSet, MultiGraph

class MGN_deq(EncodeProcessDecode):
    """MGN with deq"""
    def init_processor(self, latent_size_node, latent_size_edge):
        args=self.args
        model_fn_node = functools.partial(self._make_mlp, input_size=latent_size_node+  latent_size_edge, output_size=latent_size_node, nhid=latent_size_node)
        model_fn_edge = functools.partial(self._make_mlp, input_size=latent_size_edge+2*latent_size_node, output_size=latent_size_edge, nhid=latent_size_edge)
        self.deq_mode = self._message_passing_steps < 0
        print(f"{self.deq_mode=} {self._message_passing_steps=}")
        if args.nlayer_enc_nondeq > 0:
            message_passing_layers = [GraphNetBlock(model_fn_node, model_fn_edge) for _ in range(args.nlayer_enc_nondeq)]
            self.enc_nondeq = nn.Sequential(*message_passing_layers)
        else:
            self.enc_nondeq = nn.Identity()
        self.x2z_node = nn.Linear(latent_size_node, latent_size_node, bias=False)
        self.x2z_edge = nn.Linear(latent_size_edge, latent_size_edge, bias=False)
        if self._message_passing_steps < 0:
            self.message_passing = GraphNetBlock(model_fn_node, model_fn_edge)
            # if args.wnorm:
            apply_norm(self.message_passing)
            # self.deq = get_deq(args.deq_args)(args.deq_args)
            # self.deq_args = self.args.deq_args
            self.deq = get_deq(**vars(args.deq_args))
            print(f"debug {self.deq=} ")
        else:
            message_passing_layers = [GraphNetBlock(model_fn_node, model_fn_edge) for _ in range(self._message_passing_steps)]
            self.message_passing = nn.Sequential(*message_passing_layers)
        print(f"debug  {self.message_passing=}")

    def forward(self, graph, z=None):
        latent_graph = self.encode(graph)
        latent_graph = self.enc_nondeq(latent_graph)

        reuse = True
        if z is None:
            # z0_node = [torch.zeros_like(x_node)[None,...]]
            # z0_edge = [torch.zeros_like(i)[None,...] for i in x_edge]
            z0_node = [torch.zeros_like(latent_graph.node_features)[None,...]]
            z0_edge = [torch.zeros_like(x.features)[None,...] for ix,x in enumerate(latent_graph.edge_sets)]
            z = z0_node + z0_edge
            reuse = False
        # print(x_node, z0_node)
        # print(x_edge, z0_edge)
        # print(z0)
        # if self.args.wnorm:
        x_node = latent_graph.node_features
        x_edge = [e.features for e in latent_graph.edge_sets]
        u = [self.x2z_node(x_node)] + [self.x2z_edge(x) for x in x_edge]
        reset_norm(self.message_passing) # Reset weights for WN

        def f(*z):
            # if not self.args.all_grad:
            #     x = x.detach()
            new_node_features = z[0][0] + u[0]
            # new_edge_sets = [e._replace(features=z[ie+1][0] + self.x2z_edge(x_edge[ie])) for ie,e in enumerate(latent_graph.edge_sets)]
            new_edge_sets = [e._replace(features=z[ie+1][0] + u[ie+1]) for ie,e in enumerate(latent_graph.edge_sets)]
            g = MultiGraph(new_node_features, new_edge_sets)
            g = self.message_passing(g)
            # return [g.node_features[None,...]]+[e.features[None,...] for e in g.edge_sets]
            return [g.node_features[None,...]]+[e.features[None,...] for e in g.edge_sets]

        if self.deq_mode:
            solver_kwargs = {'f_max_iter':0} if reuse else {}
            z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
            print(info)
        else:
            z_pred = [f(*z)]

        # print(z_pred, len(z_pred))
        z_pred = z_pred[0]
        latent_graph = MultiGraph(z_pred[0][0], [e._replace(features=z_pred[ie+1][0]) for ie,e in enumerate(latent_graph.edge_sets)])
        # if self.deq_mode:
        #     # deq_func = DEQWrapper(func, z0)
        #     # print("debug 1", [i.shape for i in z0], z0_node[0].shape, z0_edge[0].shape)

        #     z_list = [each.flatten(start_dim=1) for each in z0]
        #     # print("debug 2",[i.shape for i in z_list])
        #     # print("debug 3",torch.cat(z_list, dim=1))


        #     # z_init = deq_func.list2vec(*z0)
        #     z_init = self.deq.list2vec(*z0)
        #     # print(f"{z_init.shape=} {deq_func=} {kwargs=} ")
        #     if self.args.n_deq_pretrain:
        #         z_out = z0
        #         for _ in range(np.random.randint(*self.args.n_deq_pretrain)):
        #             z_out = func(*z_out)
        #         z_out = z_out[0]
        #     else:
        #         z_out = self.deq(deq_func, z_init)[0][0][0]
        # else:
        #     latent_graph = self.message_passing(latent_graph)
        return self.decode(latent_graph)+0*graph.node_features.sum()



# def hessian_finite_diff(V, R):
#     disp = 0.005
#     H = torch.zeros(N,3, N,3)
#     en, force = V(R)
#     plus_minus= True
#     for i in range(N):
#         for k in range(3):
#             R1 = R.copy()
#             R1[i,k] += disp
#             en_ik, force_ik = V(R1)
#             if plus_minus:
#                 R2 = R.copy()
#                 R2[i,k] -= disp
#                 en_ik2, force_ik2 = V(R2)
#                 H[i,k] = -(force_ik-force_ik2)/disp/2
#             else:
#                 H[i,k] = -(force_ik-force)/disp
#     return H
