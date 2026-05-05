# Lint as: python3
# pytorch port
# ============================================================================

# # import time
# # import pickle
# import os, sys; sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MeshGraphNets import base_mp_gnn, common
from NPS_common.pt_utils import euler_angles_to_matrix
# from NPS.utility import make_optimizer, make_scheduler, count_parameters
# from meshgraphnets import dataset
# import horovod.tensorflow as hvd
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def register_args(parser):
    parser.add_argument('--core', default='MeshGraphNets')
    parser.add_argument('--nfeat_onehot', type=int, default=4, help='onehot embedding dim for node type')
    # parser.add_argument('--nfeat_edge_in', type=int, default=0, help='Note: input edge feature should be: args.dim+1+args.nfeat_edge_in')
    parser.add_argument('--nlayer_mlp_encdec', type=int, default=-1, help='MLP layer in GNN encoder/decoder')
    parser.add_argument('--euler_convention', default='ZXZ')
    parser.add_argument('--T1', type=int, default=0, help='Supply T of next step as input')
    parser.add_argument('--amr', type=str, default='pointwise_v2', help='which AMR')
    parser.add_argument('--amr_buffer', type=int, default=1, help='how many buffer grids (must be 0 or 1)')
    # parser.add_argument('--amr_eval_freq', type=int, default=1, help='Call AMR in eval every this many times (default 1)')
    parser.add_argument('--amr_threshold', type=float, default=1e-3, help='threshold to coarsen regions if values are close')
    ### Training
    # parser.add_argument('--randommesh', type=int, default=0, help='Data augmentation by generating random points and associated mesh')


def post_process_args(args):
    assert not args.channel_first, ValueError(f'GNN demands channel last')
    assert not args.RNN, ValueError('MGN is feedforward only')
    assert args.ngram == 1
    assert args.dim == 3
    assert not args.periodic
    assert not (args.amr and args.slice_op), ValueError('remeshing and slicing operators together are NOT supported yet')
    # args.feat_out_method = args.feat_out_method.split(',')
    # args.feat_out_method += [args.feat_out_method[-1]]*(args.nfeat_out-len(args.feat_out_method))
    # args.nfeat_fix = args.feat_out_method.count('fix')
    # args.nfeat_active = args.nfeat_out - args.nfeat_fix
    # args.nblocks_2wae = tuple(map(int, args.nblocks_2wae.split(',')))
    # args.nstrides_2wae = tuple(map(int, args.nstrides_2wae.split(',')))
    # if args.autoencoder == 'rev2wae':
    #     args.nfeat_autoencoder = args.nfeat_in * 2**(len(args.nstrides_2wae) * args.dim)
    # if args.nfeat_onehot is None:
    #     args.nfeat_onehot = common.NodeType.SIZE


def make_model(args):
    if args.core == 'MeshGraphNets':
        core_model_type = base_mp_gnn.EncodeProcessDecode
        learned_model = core_model_type(
                input_size_node=4+2+9, input_size_edge=args.dim+1,
                output_size=4+1+9,
                activation=args.act,
                latent_size_node=args.nfeat_hid, latent_size_edge=args.nfeat_hid_edge,
                num_layers=args.nlayer_mlp,
                nlayer_mlp_encdec=args.nlayer_mlp_encdec,
                message_passing_steps=args.n_mpassing)
    elif args.core == 'segnn':
        from GNN.segnn.segnn import SEGNN
        learned_model = SEGNN(
            input_irreps=args.input_irreps,
            hidden_irreps=args.hidden_irreps,
            output_irreps=args.output_irreps,
            edge_attr_irreps=args.edge_attr_irreps,
            node_attr_irreps=args.node_attr_irreps,
            num_layers=args.n_mpassing,
            norm=None,
            pool="avg",
            task="node",
            additional_message_irreps=None,
        )
    model = Cafe_GNN(learned_model, dim=args.dim, periodic=args.periodic, nfeat_onehot=4, euler_convention=args.euler_convention, T1=args.T1, buffer=args.amr_buffer)
    return model

def make_trainer(args, loader, model, loss, checkpoint):
    return CafeGNNTrainer(args, loader, model, loss, checkpoint)



from NPS.model.common import vector_pbc
from NPS_common.utils import grid_points, repr_simple_graph
from NPS_common.pt_utils import unique_indices

class Cafe_GNN(nn.Module):
    """Input data of 6 channels: state, 3_euler_angles, size, T, [Optionally supply T1 (T of next time step)]"""

    def __init__(self, learned_model, dim=3, periodic=False, nfeat_onehot=4, euler_convention='XYZ', T1=False, buffer=1): #nfeat_in=6, nfeat_out=4+1+9,  n_real_var=2,
        super().__init__()
        self.dim=dim
        self.periodic=periodic
        # self.n_real_var = n_real_var
        # self.nfeat_in=nfeat_in; self.nfeat_out=nfeat_out
        self._learned_model = learned_model
        self.nfeat_node_embedding = nfeat_onehot
        self.euler_convention = euler_convention
        self.T1 = T1
        indices_neighbor = torch.tensor(grid_points([2*buffer]*dim)-buffer).reshape(1,-1,dim) # all points made from [-1,0,1]
        # neighbors only, i.e. remove 0,0,0
        indices_neighbor = indices_neighbor[:, indices_neighbor[0].abs().sum(dim=1)>0]
        ## pad one space for batch index
        self.indices_neighbor = torch.cat((torch.zeros_like(indices_neighbor)[...,:1],indices_neighbor), -1).cuda() # shape(1, -1, 1+dim) # , device='CUDA'

    def get_rot(self, eulers):
        # print('eulers', eulers.shape, euler_angles_to_matrix(eulers, self.euler_convention).shape)
        return torch.where(((eulers+1).abs()<1e-6).all(dim=-1,keepdim=True), torch.zeros((*eulers.shape[:-1], 9), dtype=eulers.dtype, device=eulers.device),
          euler_angles_to_matrix(eulers, self.euler_convention).reshape(*eulers.shape[:-1],9))

    def preprocess(self, inputs):
        """Builds input graph."""
        # construct graph nodes
        node_type = nn.functional.one_hot(inputs['x'][:, 0].long(), self.nfeat_node_embedding)
        eulers = inputs['x'][:,1:4]
        node_features = torch.cat([node_type, self.get_rot(eulers), inputs['x'][:,4:]], dim=-1) #+ ([inputs['T1']] if self.T1 else [])
        # print(node_features.shape)

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


    def _update(self, inputs, per_node_network_output):
        """Features output: state (4), euler_rot (9), size(1)"""
        x_update = per_node_network_output
        cur_x = inputs['x']
        # return cur_x + x_update
        return x_update

    def predict_next(self, inputs, out):
        """Final output: state id (1), euler (3), size(1)"""
        state0 = inputs['x'][:,0].long()
        flag_inactive = state0==0
        state1 = torch.argmax(out[:,:self.nfeat_node_embedding], axis=1)
        state1[flag_inactive] = 0 # keep inactive

        field1 = out[:, -1:]
        # hard code: 0 must not be changed
        field1[flag_inactive] = 0
        field1[state1==1] = 0
        field1[state1==3] = 1
        field1[state1==2].clamp_(0, 1)

        euler1 = inputs['x'][:,1:4] ## t=0 Euler angles
        euler1[state1<=1] = -1
        flag_solidify = torch.logical_and(state0<=1, state1>=2)
        # for mushy or solid cells, inherit Euler angles unless it was liquid phase at t=0
        # print(f'debug  flag_solidify {flag_solidify.shape}, {flag_solidify} {torch.nonzero(flag_solidify).shape}')
        rot1 = out[flag_solidify, self.nfeat_node_embedding:self.nfeat_node_embedding+9]
        # print(f'debug  rot1 {rot1.shape}, {rot1}')
        ijk1 = torch.cat((inputs['batch_index'], inputs['mesh_pos'].int()), -1)[flag_solidify]
        # print(f'debug ijk1 {ijk1.shape} {ijk1} \n rot1 {rot1.shape}, {rot1} \n nbs {self.indices_neighbor.shape} {self.indices_neighbor} \n cell', inputs['aux_vars']['batch_cell_len'])
        # ijk1_nbs = ((ijk1[:,None,:] + self.indices_neighbor).reshape((-1, self.dim+1)) % inputs['aux_vars']['batch_cell_len']).long()
        ijk1_nbs = ((ijk1[:,None,:] + self.indices_neighbor) % inputs['aux_vars']['batch_cell_len']).long()
        ijk1_nbs = torch.inner(ijk1_nbs.float(), inputs['aux_vars']['ijk2int'].float()).long()
        # print(f'debug ijk {ijk1_nbs.shape} {ijk1_nbs}')
        rot0_nbs = inputs['x_dense'][ijk1_nbs, 1:4]
        # print(f'debug rot0_nbs {rot0_nbs.shape} {rot0_nbs}')
        rot0_nbs = self.get_rot(rot0_nbs)
        # print(f'debug rot0_nbs {rot0_nbs.shape} {rot0_nbs}')
        dist = (rot0_nbs - rot1[:,None]).square().sum(dim=-1)
        # print(f'debug dist {dist.shape} {dist}', torch.argmin(dist, dim=1).shape, torch.argmin(dist, dim=1))
        # idx_min = ijk1_nbs[:,torch.argmin(dist, dim=1)]
        euler1[flag_solidify] = torch.gather(rot0_nbs, 1, torch.argmin(dist, dim=1)[:,None,None].expand(-1,-1,3)).squeeze(1) #inputs['x_dense'][idx_min,1:4]
        # print(f'debug e {euler1.shape} field {field1.shape} state {state1.shape}')
        # rot0 = graph.node_features[self.nfeat_node_embedding:self.nfeat_node_embedding+9]
        # euler1 = torch.zeros((out.shape[0], 3), device=out.device, dtype=out.dtype)
        return torch.cat([state1.unsqueeze(1), euler1, field1], dim=-1)

    def forward(self, inputs, target=None, criterion=None, slice_mask=None, **kwx):
        graph = self.preprocess(inputs)
        per_node_network_output = self._learned_model(graph)
        out = self._update(inputs, per_node_network_output)
        out_frame = self.predict_next(inputs, out)
        if target is None:
            return out, out_frame
        else:
            # slice_mask = ???
            # state1 = torch.argmax(out[:,:self.nfeat_node_embedding], axis=1)
            state1 = out_frame[:,0]
            tgt = target[inputs['x_idx']]
            slice_mask = slice_mask.repeat(inputs['aux_vars']['batch']) if slice_mask is not None else torch.ones(tgt.shape[0], dtype=torch.bool, device=tgt.device)
            # loss_state = F.cross_entropy(out[:,:self.nfeat_node_embedding], tgt[:, 0].long())
            loss_state = F.cross_entropy(out[slice_mask, :self.nfeat_node_embedding], tgt[:, 0].long())
            mushy_flag  = torch.logical_and(state1 == 2, slice_mask)
            mushy_solid = torch.logical_and(state1 >= 2, slice_mask)
            loss_euler = criterion(out[mushy_solid, self.nfeat_node_embedding:self.nfeat_node_embedding+9], self.get_rot(tgt[mushy_solid, 1:4]))
            loss_field = criterion(out[mushy_flag, self.nfeat_node_embedding+9], tgt[mushy_flag, 4])
            return out, out_frame, loss_state, loss_euler, loss_field



def cafe_array2graph(frame, dim, periodic=False, device='cpu', **kwargs):
    graph = common.array2graph(frame, dim, periodic=periodic, device=device)
    try:
        del graph['node_type']
        del graph['aux_vars']['node_type_dense']
    except:
        pass
    # print(graph)
    return graph



from NPS.trainer import Trainer
class CafeGNNTrainer(Trainer):
    def __init__(self, args, *a):
        super().__init__(args, *a)
        if args.amr == '':
            from MeshGraphNets import identity_mesher
            self.mesher = identity_mesher()
            print('No AMR')
        elif args.amr == 'pointwise':
            from MeshGraphNets.amr_pointwise import amr_pointwise
            self.mesher = amr_pointwise(args.dim, periodic=args.periodic, nfeat_active=args.nfeat_active,
              nfeat_fix=args.nfeat_fix, threshold=args.amr_threshold, buffer=args.amr_buffer, device=self.device)
            print('Pointwise AMR')
        elif args.amr == 'pointwise_v2':
            from MeshGraphNets.amr_pointwise import amr_pointwise_v2 as amr_pw
            self.mesher = amr_pw(args.dim, periodic=args.periodic, flagger=lambda x: x[:,0]==2,
                threshold=args.amr_threshold, buffer=args.amr_buffer, device=self.device)
            print('Pointwise AMR V2')
        else:
            raise ValueError(f'Unknown mesher {args.amr}')
        ### NOTE: this implementation is NOT ideal! It always ignored the boundaries, even the physical boundaries (without PBC)
        args = self.args
        if ('slice' in args.data_aug) and args.slice_op:
            npass = args.n_mpassing
            slice_op = list(map(int, args.slice_op.split(',')))
            slice_size = [slice_op[i] if slice_op[i]>0 else args.frame_shape[i] for i in range(3)]
            slice_mask = np.mgrid[0:slice_size[0], 0:slice_size[1], 0:slice_size[2]]
            slice_mask = [np.logical_and(slice_mask[i]>=npass, slice_mask[i]<slice_size[i]-npass) for i in range(self.dim) if (slice_size[i]<args.frame_shape[i])]
            self.slice_mask = torch.from_numpy(np.all(slice_mask, 0).ravel()).cuda()
            # print(f'debug slice_mask {self.slice_mask.shape}, {self.slice_mask.sum()}')
        else:
            self.slice_mask = torch.ones(args.frame_shape[0]*args.frame_shape[1]*args.frame_shape[2]).bool().cuda()
            print('No slicing mask necessary')

    def train_batch(self, x, criterion=None, epoch=1):
        args = self.args
        n_in  = args.n_in
        n_out = args.n_out
        x_in = x[:, :n_in]
        # x_in = self.noise_op(x_in)
        g_in = cafe_array2graph(x_in, args.dim, periodic=args.periodic, device=x.device)
        # print('g_in', repr_simple_graph(g_in))
        g_in = self.mesher.remesh(g_in)
        # print('meshed', repr_simple_graph(g_in))
        self.optimizer.zero_grad()
        loss = 0.0
        loss_item = []
        use_teacher_forcing = False
        for di in range(n_out):
            target = x[:,n_in+di].reshape((-1,x.shape[-1]))
            # target = self.mesher.get_target(g_in, target)
            _, y, loss_state, loss_euler, loss_field = self.model(g_in, target=target, criterion=criterion, mask=self.slice_mask)
            # print(f'debug y {y.shape} tgt {target.shape} x {x[:,n_in+di].shape}')
            # new type; size; Euler angles
            loss += loss_state*1 + loss_euler*1 + loss_field*1
            loss_item.append([loss_state.item(), loss_euler.item(), loss_field.item()])
            if use_teacher_forcing:
                x_in_last = target # Teacher forcing
            else:
                x_in_last = torch.cat((y, target[g_in['x_idx']][:,-1:]), -1)
            g_in['x'] = x_in_last
            g_in['x_dense'][g_in['x_idx']] = g_in['x']
            g_in['x_dense'][:, -1] = target[:, -1]
            if di < n_out-1: g_in = self.mesher.remesh(g_in)
        loss.backward()
        self.optimizer.step()
        return loss.item() / n_out, np.mean(loss_item, 0)

    def evaluate_batch(self, x, predict_only=False):
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
        traj = []
        x_in = x[:,:n_in]
        g_in = cafe_array2graph(x_in, args.dim, periodic=args.periodic, device=x.device, T1=args.T1)
        g_in = self.mesher.remesh(g_in)
        loss_item = []
        for di in range(n_out):
            target = x[:,n_in+di].reshape((-1,x.shape[-1])) if not predict_only else None
            _, y, loss_state, loss_euler, loss_field = self.model(g_in, target=target, criterion=self.loss)
            loss_item.append([(loss_state*1 + loss_euler*1 + loss_field*1).item(), loss_state.item(), loss_euler.item(), loss_field.item()])
            x_in_last = torch.cat((y, target[g_in['x_idx']][:,-1:]), -1)
            g_in['x'] = x_in_last
            g_in['x_dense'][g_in['x_idx']] = g_in['x']
            g_in['x_dense'][:, -1] = target[:, -1]
            traj.append(self.mesher.get_last(g_in).reshape(x.shape[0],*x.shape[2:2+self.args.dim],-1).detach())
            if di < n_out-1: g_in = self.mesher.remesh(g_in)
        return torch.stack(traj, 1), np.mean(loss_item, 0)


