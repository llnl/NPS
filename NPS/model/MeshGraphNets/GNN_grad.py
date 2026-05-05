# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for graph neural network."""

import torch
import torch.nn as nn
import torch_scatter

from . import GNN

def register_args(parser):
    parser.add_argument('--n_grad_in', type=str, default="1,4", help="dim of non-x and x (to differentiate WRT)")
    parser.add_argument('--n_grad_gt', type=str, default="4,1,2", help="in GT: dim of grad, ignored sum, components to sum")
    parser.add_argument('--UpDnSymm', type=int, default=0, help="Spin up/down symmetry")
    parser.add_argument('--diag_only', type=int, default=0, help="off diagonal n_{s1,s2}")
    parser.add_argument('--return_sum', type=int, default=1, help="sum up energy to include in return values")

def post_process_args(args):
    args.n_grad_in = list(map(int, args.n_grad_in.split(",")))
    args.n_grad_gt = list(map(int, args.n_grad_gt.split(",")))
    # args.n_tot_out = list(map(int, args.n_tot_out.split(",")))


class GNN_grad(GNN.GNN):
    """Model with derivative as an output"""

    def forward(self, inputs, target=None, criterion=nn.functional.mse_loss, **kwx):
        graph = self.preprocess(inputs, is_training=False)
        r = self.args
        non_x, x = torch.split(graph.node_features, r.n_grad_in, dim=1)
        if r.diag_only:
            x = x[:, ::3]
        x.requires_grad_(True)
        with torch.enable_grad():
            graph.node_features = torch.cat((non_x, x),-1)
            node_output = self._learned_model(graph)
            if r.UpDnSymm:
                graph.node_features = torch.cat((non_x, torch.flip(x, [1])),-1)
                node_output = (node_output + self._learned_model(graph))/2
            ygrad = torch.autograd.grad(
                    [node_output.sum()],
                    x,
                    create_graph=True,#self.training,  # needed to allow gradients of this output during training
                )[0]
            if r.diag_only:
                ygrad = torch.cat((ygrad[:,:1], ygrad*0, ygrad[:,-1:]), -1)
        # x.requires_grad_(False)
        # if self.nfeat_out_global > 0:
        #     global_output = torch_scatter.scatter_mean(node_output[:, :self.nfeat_out_global], inputs.batch, dim=0)
        #     node_output = node_output[:, self.nfeat_out_global:]
        # else:
        #     global_output = None
        if r.return_sum:
            out = torch.cat((ygrad, node_output.sum(dim=-1, keepdim=True), node_output), dim=-1)
        else:
            out = torch.cat((ygrad, node_output), dim=-1)
        if target is not None:
            grad_gt, _, component_gt = torch.split(target, r.n_grad_gt, dim=1)
        # if self.training:
            # print(f"{ygrad.shape=} {grad_gt.shape=} {node_output.shape=} {component_gt.shape=} {non_x.shape=} {x.shape=} {graph.node_features.shape=} {node_output.shape=}")
            loss_grad =  criterion(ygrad, grad_gt)
            loss_tot  =  criterion(node_output, component_gt)
            return out, [loss_grad, loss_tot]
        else:
            return out

