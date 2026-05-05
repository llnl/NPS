"""
 Simple point-wise AMR using PyTorch
"""
__author__ = 'Fei Zhou'

import numpy as np
import torch
# np.set_printoptions(edgeitems=30, linewidth=200, formatter=dict(float=lambda x: "%.3g" % x))
# import tensorflow.compat.v1 as tf
from NPS_common.utils import grid_points#, digits2int, linear_interp_coeff
from NPS_common.pt_utils import unique_indices
# from NPS_common.tf_utils import int2digits

class amr_pointwise:
    def __init__(self, dim, periodic=True, nfeat_active=1, nfeat_fix=0, threshold=1e-3, buffer=2, device='cpu'):
        self.dim = dim
        self.periodic = periodic
        self.nfeat_active = nfeat_active
        self.nfeat_fix = nfeat_fix
        self.threshold = threshold
        self.buffer = buffer
        indices_neighbor = torch.tensor(grid_points([2*buffer]*dim)-buffer, device=device).reshape(1,-1,dim) # all points made from [-1,0,1]
        ## pad one space for batch index
        self.indices_neighbor = torch.cat((torch.zeros_like(indices_neighbor)[...,:1],indices_neighbor), -1) # shape(1, -1, 1+dim)
        # self.device = device

    def remesh(self, g_in, x=None):
        assert g_in['xtime'].shape[-1] == 1, ValueError(f'ERROR: ngram>1 NOT supported in amr_pointwise')
        if x is None: x = g_in['xtime'][...,-1]
        # print(f"debug start x sum {g_in['xtime'].shape} {g_in['xtime'][...,0,-1].sum()} {g_in['xtime'][...,-1,-1].sum()}")
        # print('ijk2int', g_in['aux_vars']['ijk2int'])
        graph = {'lattice':g_in['lattice'], 'inv_lattice':g_in['inv_lattice'],
          'aux_vars':g_in['aux_vars'], 'x_fix':g_in['x_fix']}
        # print(f'debug ', self.nfeat_fix, graph['x_fix'] )
        if (self.nfeat_fix > 0) and (g_in['x_fix'] is None):
            graph['x_fix'] = x[..., self.nfeat_active:]
        x_dense = torch.zeros((g_in['aux_vars']['n_dense'], self.nfeat_active), device=x.device)
        x_dense[g_in['x_idx']] = x[...,:self.nfeat_active]
        if self.nfeat_fix > 0:
            x_dense = torch.cat((x_dense, graph['x_fix']), -1)
        # x_dense = torch.zeros((g_in['aux_vars']['batch'], *g_in['aux_vars']['shape'], x.shape[-1]), device=x.device)
        idx_cur = torch.cat((g_in['batch_index'], g_in['mesh_pos'].int()), -1)
        # print(f"debug point x sum {x.shape} {x[...,0].sum()} {x[...,-1].sum()} x_dense {x_dense[...,0].sum()} {x_dense[...,-1].sum()}")
        point_flag = (x[...,:self.nfeat_active].abs().sum(dim=-1) > self.threshold)
        # idx_new = ((idx_cur[point_flag][:,None,:] + self.indices_neighbor).reshape((-1, self.dim+1)) % g_in['aux_vars']['batch_cell_len']).long()
        idx_new = (idx_cur[point_flag][:,None,:] + self.indices_neighbor).reshape((-1, self.dim+1))
        if self.periodic:
            idx_new = (idx_new % g_in['aux_vars']['batch_cell_len']).long()
        else:
            idx_new = idx_new[torch.logical_and(torch.all(idx_new>=0 ,dim=1), torch.all(idx_new<g_in['aux_vars']['batch_cell_len'] ,dim=1))].long()
        # print(f'debug point_flag {point_flag.shape} {point_flag.sum()} idx {idx_new.shape} {idx_new}')
        # print(idx_new.dtype, idx_new.shape, g_in['aux_vars']['idx_fac'].dtype, g_in['aux_vars']['idx_fac'].shape)
        idx_new = torch.inner(idx_new.float(), g_in['aux_vars']['ijk2int'].float()).long()
        # print(f'debug idx {idx_new.shape} {idx_new}')
        # idx_new = torch.unique(idx_new, dim=0)
        idx_new = unique_indices(idx_new, g_in['aux_vars']['n_dense'])
        # print(f'debug idx {idx_new.shape} {idx_new} x {x_dense[idx_new,0].sum()} {x_dense[idx_new,-1].sum()} x_dense {x_dense[:,0].sum()} {x_dense[:,-1].sum()} ')
        # print( point_flag.int().sum(), idx_cur.shape, idx_new.shape, 'flags', point_flag)
        # print(idx_new.shape, idx_new)
        mask_all = torch.zeros(g_in['aux_vars']['n_dense'], dtype=torch.bool, device=x.device)
        mask_all[idx_new] = 1
        index_final = torch.cumsum(mask_all, 0)-1
        mask_edge = mask_all[g_in['aux_vars']['edge_index_dense'][:,0]] * mask_all[g_in['aux_vars']['edge_index_dense'][:,1]]

        graph['mesh_pos'] = g_in['aux_vars']['mesh_pos_dense'][idx_new]
        graph['node_type'] = g_in['aux_vars']['node_type_dense'][idx_new]
        graph['edge_index'] = index_final[g_in['aux_vars']['edge_index_dense'][mask_edge.bool()].reshape(-1)].reshape((-1,2))
        # print(f'debug edge idx ', g_in['aux_vars']['edge_index_dense'].shape, edge_new.shape, 'non zero', torch.nonzero(edge_new).shape, 'new idx', graph['edge_index'].shape)
        graph['batch_index'] = g_in['aux_vars']['batch_index_dense'][idx_new]
        graph['x_dense'] = x_dense
        graph['x'] = x_dense[idx_new]
        graph['xtime'] = graph['x'].reshape((-1,g_in['xtime'].shape[-2],g_in['xtime'].shape[-1]))
        graph['x_idx'] = idx_new
        # print(f"debug end   x sum {graph['xtime'].shape} {graph['xtime'][...,0,-1].sum()} {graph['xtime'][...,-1,-1].sum()}")
        return graph

    def get_last(self, g_in):
        return g_in['x_dense']

    def get_target(self, g_in, tgt):
        return tgt[g_in['x_idx']]


class amr_pointwise_v2:
    def __init__(self, dim, periodic=False, flagger=lambda x: x[:,:1].abs().sum(dim=-1), nfeat_fix=0, threshold=1e-3, buffer=2, device='cpu'):
        self.dim = dim
        self.periodic = periodic
        self.flagger = flagger
        # self.nfeat_fix = nfeat_fix
        self.threshold = threshold
        self.buffer = buffer
        indices_neighbor = torch.tensor(grid_points([2*buffer]*dim)-buffer, device=device).reshape(1,-1,dim) # all points made from [-1,0,1]
        ## pad one space for batch index
        self.indices_neighbor = torch.cat((torch.zeros_like(indices_neighbor)[...,:1],indices_neighbor), -1) # shape(1, -1, 1+dim)
        # self.device = device

    def remesh(self, g_in, x=None, x_dense=None, g_global=None):
        if x is None: x = g_in['x']
        point_flag = (self.flagger(x) > self.threshold)
        if g_global is None: g_global = g_in['aux_vars']
        N_v_all = g_global['n_dense']
        # print(g_global)
        # print(list(g_global.keys()))
        if x_dense is None: x_dense = g_in['x_dense']
        # print(f"debug start x sum {g_in['xtime'].shape} {g_in['xtime'][...,0,-1].sum()} {g_in['xtime'][...,-1,-1].sum()}")
        # print('ijk2int', g_in['aux_vars']['ijk2int'])
        graph = {'lattice':g_in['lattice'], 'inv_lattice':g_in['inv_lattice'],
          'aux_vars':g_in['aux_vars']}#, 'x_fix':g_in['x_fix']}
        # print(f'debug ', self.nfeat_fix, graph['x_fix'] )
        # if (self.nfeat_fix > 0) and (g_in['x_fix'] is None):
        #     graph['x_fix'] = x[..., self.nfeat_active:]
        # x_dense = torch.zeros((g_in['aux_vars']['n_dense'], self.nfeat_active), device=x.device)
        # x_dense[g_in['x_idx']] = x[...,:self.nfeat_active]
        # if self.nfeat_fix > 0:
        #     x_dense = torch.cat((x_dense, graph['x_fix']), -1)
        # x_dense = torch.zeros((g_in['aux_vars']['batch'], *g_in['aux_vars']['shape'], x.shape[-1]), device=x.device)
        idx_cur = torch.cat((g_in['batch_index'], g_in['mesh_pos'].int()), -1)
        # print(f"debug point x sum {x.shape} {x[...,0].sum()} {x[...,-1].sum()} x_dense {x_dense[...,0].sum()} {x_dense[...,-1].sum()}")
        # print(f'debug idx_cur {idx_cur.shape} flat {point_flag.shape}')
        idx_new = (idx_cur[point_flag][:,None,:] + self.indices_neighbor).reshape((-1, self.dim+1))
        if self.periodic:
            idx_new = (idx_new % g_in['aux_vars']['batch_cell_len']).long()
        else:
            idx_new = idx_new[torch.logical_and(torch.all(idx_new>=0 ,dim=1), torch.all(idx_new<g_in['aux_vars']['batch_cell_len'] ,dim=1))].long()
        # print(f'debug point_flag {point_flag.shape} {point_flag.sum()} idx {idx_new.shape} {idx_new}')
        # print(idx_new.dtype, idx_new.shape, g_in['aux_vars']['idx_fac'].dtype, g_in['aux_vars']['idx_fac'].shape)
        idx_new = torch.inner(idx_new.float(), g_in['aux_vars']['ijk2int'].float()).long()
        # print(f'debug idx {idx_new.shape} {idx_new}')
        idx_new = unique_indices(idx_new, N_v_all)
        # print(f'debug idx {idx_new.shape} {idx_new} x {x_dense[idx_new,0].sum()} {x_dense[idx_new,-1].sum()} x_dense {x_dense[:,0].sum()} {x_dense[:,-1].sum()} ')
        # print( point_flag.int().sum(), idx_cur.shape, idx_new.shape, 'flags', point_flag)
        # print(idx_new.shape, idx_new)
        mask_all = torch.zeros(N_v_all, dtype=torch.bool, device=x.device)
        mask_all[idx_new] = 1
        index_final = torch.cumsum(mask_all, 0)-1
        mask_edge = mask_all[g_global['edge_index_dense'][:,0]] * mask_all[g_global['edge_index_dense'][:,1]]

        graph['mesh_pos'] = g_global['mesh_pos_dense'][idx_new]
        # graph['node_type'] = g_in['node_type_dense'][idx_new]
        graph['edge_index'] = index_final[g_global['edge_index_dense'][mask_edge.bool()].reshape(-1)].reshape((-1,2))
        # print(f'debug edge idx ', g_in['aux_vars']['edge_index_dense'].shape, edge_new.shape, 'non zero', torch.nonzero(edge_new).shape, 'new idx', graph['edge_index'].shape)
        graph['batch_index'] = g_global['batch_index_dense'][idx_new]
        graph['x_dense'] = x_dense
        # print(f'x d {x_dense} idx {idx_new}')
        graph['x'] = x_dense[idx_new]
        graph['x_idx'] = idx_new
        # print(f"debug end   x sum {graph['xtime'].shape} {graph['xtime'][...,0,-1].sum()} {graph['xtime'][...,-1,-1].sum()}")
        return graph

    def get_last(self, g_in):
        return g_in['x_dense']

    def get_target(self, g_in, tgt):
        return tgt[g_in['x_idx']]


if __name__ == '__main__':
    from MeshGraphNets import common
    from NPS_common.utils import load_array_auto
    import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; import seaborn as sns;
    all = load_array_auto('/usr/WS1/amsdnn/2D_forest/dataset3/valid.sp.npz')

    # print('testing updating mesh values')
    # frame = torch.from_numpy(all[1:2, 30:31,..., ::2])
    # mesher = amr_pointwise(2, nfeat_active=1, nfeat_fix=1)
    # g_in = common.array2graph(frame, 2, time_dim=True)
    # # print(g_in.__class__, list(g_in.keys()))
    # fig, ax = plt.subplots(1, 4)
    # for i in range(5):
    #     g_in = mesher.remesh(g_in, torch.from_numpy(all[1:2, 31+i,..., ::2]).reshape((-1,2))[g_in['x_idx']])
    #     pos = g_in['mesh_pos']
    #     ax[0].scatter(*(pos.T), s=2)
    #     # print('debug', g_in['x_dense'].shape, g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0].shape, g_in['x_dense'].__class__, np.nonzero(g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0]))
    #     ax[1].scatter(*(g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0].nonzero(as_tuple=True)), s=2)
    #     ax[2].scatter(*(g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,1].nonzero(as_tuple=True)), s=2)
    #     #if g_in['x_fix'] is not None: 
    #     ax[3].scatter(*(g_in['x_fix'].reshape(*g_in['aux_vars']['shape']).nonzero(as_tuple=True)), s=2)
    # plt.show()

    for nfix in (0, 1):
        for threshold in (-1, 1e-3):
            fig, ax = plt.subplots(1, 8)
            for batch in (1, 2):
                print(f'testing nfix = {nfix} threshold = {threshold} batch = {batch}')
                frame = torch.from_numpy(all[1:1+batch, 300:301,..., :1+2*nfix:2])
                mesher = amr_pointwise(2, nfeat_active=1, nfeat_fix=nfix, threshold=threshold)
                g_in = common.array2graph(frame, 2, time_dim=True)
                ifig = (batch-1)*4
                for i in range(3):
                    g_in = mesher.remesh(g_in, torch.from_numpy(all[1:1+batch, 301+i,..., ::2]).reshape((-1,2))[g_in['x_idx']])
                    pos = g_in['mesh_pos']
                    ax[0+ifig].scatter(*(pos.T), s=2)
                    # print('debug', g_in['x_dense'].shape, g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0].shape, g_in['x_dense'].__class__, np.nonzero(g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0]))
                    ax[1+ifig].scatter(*(g_in['x_dense'].reshape(batch, *g_in['aux_vars']['shape'],1+nfix)[...,0].nonzero(as_tuple=True))[1:], s=2)
                    ax[2+ifig].scatter(*(g_in['x_dense'].reshape(batch, *g_in['aux_vars']['shape'],1+nfix)[...,-1].nonzero(as_tuple=True))[1:], s=2)
                    #if g_in['x_fix'] is not None: 
                    if nfix>0: ax[3+ifig].scatter(*(g_in['x_fix'].reshape(batch, *g_in['aux_vars']['shape'])[:,1:].nonzero(as_tuple=True))[1:], s=2)
                    # print(f'debug time {i} x', g_in['x_idx'].shape, g_in['x_idx'])
            plt.show()


    # nfix=0
    # print('testing batch')
    # fig, ax = plt.subplots(1, 8)
    # for batch in (1, 2):
    #     print(f'batch = {batch}')
    #     frame = torch.from_numpy(all[1:1+batch, 300:301,..., :1])
    #     mesher = amr_pointwise(2, nfeat_active=1, nfeat_fix=0, threshold=-1)
    #     g_in = common.array2graph(frame, 2, time_dim=True)
    #     ifig = (batch-1)*4
    #     for i in range(3):
    #         g_in = mesher.remesh(g_in)#, torch.from_numpy(all[1:1+batch, 301+i,..., ::2]).reshape((-1,2))[g_in['x_idx']])
    #         pos = g_in['mesh_pos']
    #         ax[0+ifig].scatter(*(pos.T), s=2)
    #         # print('debug', g_in['x_dense'].shape, g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0].shape, g_in['x_dense'].__class__, np.nonzero(g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0]))
    #         ax[1+ifig].scatter(*(g_in['x_dense'].reshape(batch, *g_in['aux_vars']['shape'],1+nfix)[...,0].nonzero(as_tuple=True))[1:], s=2)
    #         ax[2+ifig].scatter(*(g_in['x_dense'].reshape(batch, *g_in['aux_vars']['shape'],1+nfix)[...,-1].nonzero(as_tuple=True))[1:], s=2)
    #         #if g_in['x_fix'] is not None: 
    #         if nfix>0: ax[3+ifig].scatter(*(g_in['x_fix'].reshape(batch, *g_in['aux_vars']['shape'])[:,1:].nonzero(as_tuple=True))[1:], s=2)
    #         print(f'debug time {i} x', g_in['x_idx'].shape, g_in['x_idx'])
    # plt.show()

    # print('testing batch')
    # fig, ax = plt.subplots(1, 8)
    # for batch in (1, 2):
    #     print(f'batch = {batch}')
    #     frame = torch.from_numpy(all[1:1+batch, 300:301,..., ::2])
    #     mesher = amr_pointwise(2, nfeat_active=1, nfeat_fix=1)
    #     g_in = common.array2graph(frame, 2, time_dim=True)
    #     ifig = (batch-1)*4
    #     for i in range(3):
    #         g_in = mesher.remesh(g_in)#, torch.from_numpy(all[1:1+batch, 301+i,..., ::2]).reshape((-1,2))[g_in['x_idx']])
    #         pos = g_in['mesh_pos']
    #         ax[0+ifig].scatter(*(pos.T), s=2)
    #         # print('debug', g_in['x_dense'].shape, g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0].shape, g_in['x_dense'].__class__, np.nonzero(g_in['x_dense'].reshape(*g_in['aux_vars']['shape'],2)[...,0]))
    #         ax[1+ifig].scatter(*(g_in['x_dense'].reshape(batch, *g_in['aux_vars']['shape'],2)[...,0].nonzero(as_tuple=True))[1:], s=2)
    #         ax[2+ifig].scatter(*(g_in['x_dense'].reshape(batch, *g_in['aux_vars']['shape'],2)[...,1].nonzero(as_tuple=True))[1:], s=2)
    #         #if g_in['x_fix'] is not None: 
    #         ax[3+ifig].scatter(*(g_in['x_fix'].reshape(batch, *g_in['aux_vars']['shape'])[:,1:].nonzero(as_tuple=True))[1:], s=2)
    #         print(f'debug time {i} x', g_in['x_idx'].shape, g_in['x_idx'])
    # plt.show()
