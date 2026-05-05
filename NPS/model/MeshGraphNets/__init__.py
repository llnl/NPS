# Lint as: python3
# pytorch port
# ============================================================================

# import time
# import pickle
import os, sys; sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import torch.nn as nn

# from MeshGraphNets.option import args
# tf.logging.set_verbosity(tf.logging.ERROR)
# from meshgraphnets import cfd_eval
# from meshgraphnets import amr_eval
from NPS.model.MeshGraphNets import NPS_model
from NPS.model.MeshGraphNets import base_mp_gnn, common
# from NPS.utility import make_optimizer, make_scheduler, count_parameters
# from meshgraphnets import dataset
# import horovod.tensorflow as hvd
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def register_args(parser):
    parser.add_argument('--gnnmodel', type=str, default='NPS', help='Select model to run.')
    parser.add_argument('--core', type=str, default='base_mp_gnn', help='core model')
    parser.add_argument('--unique_op', action='store_true', default=False, help='Apply tf.unique operator among edges. Turn off to speed up but be sure edges are unique')
    parser.add_argument('--feat_out_method', type=str, default='res', help='how each output channels are processed, e.g. "res,fix". Choices= res: NN(x)+x; id: NN(x); sigmoid: sigmoid(NN(x)); fix: x')
    parser.add_argument('--nfeat_onehot', type=int, default=None, help='onehot embedding dim for node type')
    parser.add_argument('--nfeat_edge_in', type=int, default=0, help='Note: input edge feature should be: args.dim+1+args.nfeat_edge_in')
    # reversible 2-way AE
    parser.add_argument('--autoencoder', type=str, default='rev2wae', help='autoencoder')
    parser.add_argument('--g_dim', type=int, default=512,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--nblocks_2wae', type=str, default='2,2,2', help='number of blocks in 2way autoencoder')
    parser.add_argument('--nstrides_2wae', type=str, default='1,2,2', help='number of strides in 2way autoencoder')
    parser.add_argument('--nfeat_autoencoder', type=int, default=64, help='bottleneck dim of autoencoder')
    parser.add_argument('--nlayer_mlp_encdec', type=int, default=-1, help='MLP layer in GNN encoder/decoder')
    # parser.add_argument('--evaluator', type=str, default='cfd_eval', help='Select rollout method.')
    ### Mesh
    # parser.add_argument('--amr_N', type=int, default=64, help='system size, i.e. how many (fine) grids totally')
    # parser.add_argument('--amr_N1', type=int, default=1, help='how many (fine) grids to bin into one, 1 to disable')
    parser.add_argument('--amr', type=str, default='', help='which AMR')
    parser.add_argument('--amr_buffer', type=int, default=1, help='how many buffer grids (must be 0 or 1)')
    parser.add_argument('--amr_eval_freq', type=int, default=1, help='Call AMR in eval every this many times (default 1)')
    parser.add_argument('--amr_threshold', type=float, default=1e-3, help='threshold to coarsen regions if values are close')
    ### Training
    # parser.add_argument('--randommesh', type=int, default=0, help='Data augmentation by generating random points and associated mesh')


def post_process_args(args):
    assert not args.channel_first, ValueError(f'GNN demands channel last')
    assert not args.RNN, ValueError('MGN is feedforward only')
    args.feat_out_method = args.feat_out_method.split(',')
    args.feat_out_method += [args.feat_out_method[-1]]*(args.nfeat_out-len(args.feat_out_method))
    args.nfeat_fix = args.feat_out_method.count('fix')
    args.nfeat_active = args.nfeat_out - args.nfeat_fix
    args.nblocks_2wae = tuple(map(int, args.nblocks_2wae.split(',')))
    args.nstrides_2wae = tuple(map(int, args.nstrides_2wae.split(',')))
    if args.autoencoder == 'rev2wae':
        args.nfeat_autoencoder = args.nfeat_in * 2**(len(args.nstrides_2wae) * args.dim)
    if args.nfeat_onehot is None:
        args.nfeat_onehot = common.NodeType.SIZE


def make_model(args):
    if args.gnnmodel == 'equi_gnn':
        from MeshGraphNets import equi_gnn
        return equi_gnn.make_model(args)
    if args.gnnmodel == 'NPS_autoencoder':
        input_size_node = args.nfeat_autoencoder
        output_size = args.nfeat_autoencoder
    elif args.gnnmodel == 'GNN_mobility':
        input_size_node = args.nfeat_in + args.nfeat_onehot
        output_size = args.nfeat_out*args.dim
    else:
        input_size_node = args.nfeat_in*args.ngram + args.nfeat_onehot
        output_size = args.nfeat_out + args.nfeat_out_global
    if args.core == 'graph_gradient_model':
        from MeshGraphNets import graph_gradient_model
        core_model_type = graph_gradient_model.GNGradientNet
    elif args.core == 'diffusion':
        from MeshGraphNets import diffusion_gnn
        core_model_type = diffusion_gnn.EncodeProcessDecodeDiffusion
    # elif args.core == 'processor_only':
    #     from MeshGraphNets import NPS_autoencoder
    #     core_model_type = NPS_autoencoder.processor_only
    elif args.core == 'MGN_deq':
        from .MGN_deq import MGN_deq
        core_model_type = MGN_deq
    else:
        core_model_type = base_mp_gnn.EncodeProcessDecode
    learned_model = core_model_type(
            input_size_node=input_size_node, input_size_edge=args.dim+1+args.nfeat_edge_in,
            output_size=output_size,
            activation=args.act,
            latent_size_node=args.nfeat_hid, latent_size_edge=args.nfeat_hid_edge,
            num_layers=args.nlayer_mlp,
            nlayer_mlp_encdec=args.nlayer_mlp_encdec,
            dropout=args.dropout,
            args=args,
            message_passing_steps=args.n_mpassing)
    if args.gnnmodel == 'NPS':
        model = NPS_model.BaseNPSGNNModel(learned_model, 
            dim=args.dim, periodic=args.periodic, nfeat_in=args.nfeat_in,
            feat_out_method=args.feat_out_method,
            nfeat_out=args.nfeat_out, unique_op=args.unique_op)
    elif args.gnnmodel == 'NPS_autoencoder':
        from NPS.model.MeshGraphNets import NPS_autoencoder
        if args.autoencoder == 'rev2wae':
            from NPS.model.CrevNet.crevnet import autoencoder as AE
            assert args.nfeat_autoencoder == args.nfeat_in * 2**(len(args.nstrides_2wae) * args.dim)
            autoencoder = AE(nBlocks=args.nblocks_2wae, nStrides=args.nstrides_2wae,
                            nChannels=None, init_ds=2,
                            dropout_rate=args.dropout, affineBN=True, in_shape=(args.nfeat_in,)+args.frame_shape,
                            mult=4, dim=args.dim, periodic=args.periodic)
        else:
            raise ValueError(f'Unknown autoencoder type {args.autoencoder}')
        model = NPS_autoencoder.NPS_autoencoder(autoencoder, learned_model,
            dim=args.dim, periodic=args.periodic, nfeat_in=args.nfeat_in,
            feat_out_method=args.feat_out_method,
            nfeat_out=args.nfeat_out, unique_op=args.unique_op)
    elif args.gnnmodel == 'GNN':
        from . import GNN
        model = GNN.GNN(learned_model, 
            nfeat_in=args.nfeat_in,
            nfeat_out=args.nfeat_out, nfeat_out_global=args.nfeat_out_global)
    elif args.gnnmodel == "GNN_grad":
        from . import GNN_grad
        model = GNN_grad.GNN_grad(learned_model,
            nfeat_in=args.nfeat_in,
            nfeat_out=args.nfeat_out, nfeat_out_global=args.nfeat_out_global, args=args)
    elif args.gnnmodel == 'GNN_mobility':
        from MeshGraphNets import GNN_mobility 
        model = GNN_mobility.GNN_mobility(learned_model, 
            nfeat_in=args.nfeat_in,
            nfeat_out=args.nfeat_out, nfeat_out_global=args.nfeat_out_global)
    else:
        raise ValueError(f'unknown model {args.gnnmodel}')
    return model


def make_trainer(args, loader, model, loss, checkpoint):
    return MGNTrainer(args, loader, model, loss, checkpoint)

from NPS.trainer_AE_GNN import Trainer
class MGNTrainer(Trainer):
    def __init__(self, args, *a):
        super().__init__(args, *a)
        if args.amr == '':
            self.mesher = identity_mesher()
        elif args.amr == 'pointwise':
            from MeshGraphNets.amr_pointwise import amr_pointwise
            self.mesher = amr_pointwise(args.dim, periodic=args.periodic, nfeat_active=args.nfeat_active,
              nfeat_fix=args.nfeat_fix, threshold=args.amr_threshold, buffer=args.amr_buffer, device=self.device)
        
        
        if args.infer_mode == 'original':
            self.evaluate_batch = self._evaluate_batch_original
        elif args.infer_mode == 'optimize':
            self.evaluate_batch = self._evaluate_batch_optimize
        else:
            raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

    def train_batch(self, x, criterion=None, epoch=1):
        args = self.args
        n_in  = args.n_in
        n_out = args.n_out
        ngram = args.ngram
        x_in = x[:,n_in-ngram:n_in]
        x_in = self.augment_op(x_in)
        if not callable(criterion):
            criterion = self.loss

        g_in = common.array2graph(x_in, args.dim, periodic=args.periodic, device=x.device, time_dim=True)
        g_in = self.mesher.remesh(g_in)
        self.optimizer.zero_grad()
        loss = 0.0
        use_teacher_forcing = False
        for di in range(n_out):
            y = self.model(g_in)
            target = x[:,n_in+di].reshape((-1,x.shape[-1]))
            target = self.mesher.get_target(g_in, target)
            # print(f'debug y {y.shape} tgt {target.shape} x {x[:,n_in+di].shape}')
            loss += criterion(y, target)
            if use_teacher_forcing:
                x_in_last = target # Teacher forcing
            else:
                x_in_last = y
            g_in['xtime'] = torch.cat((g_in['xtime'][...,1:],x_in_last[...,None]),-1)
            g_in['x'] = g_in['xtime'].reshape((-1,g_in['xtime'].shape[-1]*g_in['xtime'].shape[-2]))
            if di < n_out-1: g_in = self.mesher.remesh(g_in)
        loss.backward()
        self.optimizer.step()
        return loss.item() / n_out

    def evaluate_batch(self, x, predict_only=False): ### original
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
	
        ngram = args.ngram
        traj = []
        x_in = x[:,n_in-ngram:n_in]
            #     traj.append(x_in_last.reshape(x.shape[0],*x.shape[2:2+self.args.dim],-1).detach())
        return torch.stack(traj, 1)

    def _evaluate_batch_original(self, x, predict_only=False): ### original
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
	
        ngram = args.ngram
        traj = []
        x_in = x[:,n_in-ngram:n_in]
        g_in = common.array2graph(x_in, args.dim, periodic=args.periodic, device=x.device, time_dim=True)
        g_in = self.mesher.remesh(g_in)
        for di in range(n_out):
        # for _ in range(20): # for (64,64,64) and out 20 step to 
            y = self.model(g_in)
            x_in_last = y
            g_in['xtime'] = torch.cat((g_in['xtime'][...,1:],x_in_last[...,None]),-1)
            g_in['x'] = g_in['xtime'].reshape((-1,g_in['xtime'].shape[-1]*g_in['xtime'].shape[-2]))
            g_in = self.mesher.remesh(g_in)
            traj.append(self.mesher.get_last(g_in).reshape(x.shape[0],*x.shape[2:2+self.args.dim],-1).detach())
            # if self.mesher is not None:
            #     traj.append(g_in['x_dense'].reshape(x.shape[0],*x.shape[2:2+self.args.dim],-1).detach())
            # else:
            #     traj.append(x_in_last.reshape(x.shape[0],*x.shape[2:2+self.args.dim],-1).detach())
        return torch.stack(traj, 1)

    def _evaluate_batch_optimize(self, x, predict_only=False):
        args = self.args
        n_in = args.n_in_test
        n_out = args.n_out_test
        ngram = args.ngram

        model = self.model.module if hasattr(self.model, "module") else self.model

        # Step 1: Prepare input
        x_in = x[:, n_in - ngram:n_in]  # [B, T_in, D, H, W, C]
        # print(f"[DEBUG] ngram size: {ngram}")
        B, T, *spatial, C = x_in.shape  # spatial = [D, H, W]
        assert T == ngram, f"Expected ngram={ngram}, got T={T}"

        x_in = x_in.permute(0, 5, 1, 2, 3, 4)  # [B, C, T, D, H, W]
        if C == 1:
            x_in = x_in[:, 0]  # [B, T, D, H, W] if single-channel

        # Step 2: Encode
        latent_encoded = model.autoencoder(x_in)  # → tuple of tensors
        latent_encoded_cat = torch.cat(latent_encoded, dim=1)  # [B, latent_dim, T, D, H, W]

        latent_encoded_cat = latent_encoded_cat.permute(model.ch_last_list)[:, None]  # [B, 1, D, H, W, C']
        latent_spatial = latent_encoded_cat.shape[2:5]  # [D, H, W]
        print(f"[DEBUG] latent_encoded_cat shape: {latent_encoded_cat.shape}")

        # Step 3: Convert to graph
        g_in = common.array2graph(latent_encoded_cat, args.dim, periodic=args.periodic, device=x.device, time_dim=True)
        g_in = self.mesher.remesh(g_in)

        # Step 4: GNN multiple times to update state
        for _ in range(n_out):
        # for _ in range(20):     # for (64,64,64) and out 20 step to 
            graph = model.preprocess(g_in, is_training=False)
            y_encoded = model._learned_model(graph)  # per-node prediction
            g_in['xtime'] = torch.cat((g_in['xtime'][..., 1:], y_encoded[..., None]), dim=-1)
            g_in['x'] = g_in['xtime'].reshape((-1, g_in['xtime'].shape[-1] * g_in['xtime'].shape[-2]))
            g_in = self.mesher.remesh(g_in)

        # Step 5: Decode final latent
        final_encoded = self.mesher.get_last(g_in)
        final_encoded = final_encoded.reshape(B, *latent_spatial, -1).permute(model.ch_first_list)  # [B, C, D, H, W]
        print(f"[DEBUG] final_encoded shape for decode: {final_encoded.shape}")

        # Decode using full latent
        C = final_encoded.shape[1]
        decoded = model.autoencoder(
            (final_encoded[:, :C // 2], final_encoded[:, C // 2:]),
            is_predict=False
        )
        decoded = decoded.permute(model.ch_last_list)  # [B, D, H, W, C]

        # Step 6: Final reshape
        return decoded.reshape(B, 1, *spatial, -1)  # [B, T_out=1, D, H, W, C]



class identity_mesher:
    def __init__(self, *arg):
        pass

    def remesh(self, g_in, x=None):
        assert x is None, 'DO NOT pass updated x to dummy mesher'
        return g_in

    def get_last(self, g_in):
        if 'xtime' in g_in:
            return g_in['xtime'][...,-1]
        else:
            return g_in['x']

    def get_target(self, g_in, tgt):
        return tgt


# def make_mesher(args):
#     if args.amr_N1 > 1:
#         from meshgraphnets import amr
#         print('''************* WARNING *************
#         The present AMR implementation assumes an input field on a cubic grid of size amr_N
#         ordered naturally. Make sure your dataset follows this convention''')
#         mesher = amr.amr_state_variables(args.dim, [args.amr_N]*args.dim,
#             [args.amr_N//args.amr_N1]*args.dim,
#             torch.zeros([args.amr_N**args.dim,1],dtype=torch.float32),
#             refine_threshold=args.amr_threshold, buffer=args.amr_buffer, eval_freq=args.amr_eval_freq)
#     else:
#         mesher = None
#     return mesher

