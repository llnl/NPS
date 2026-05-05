# Lint as: python3
# pytorch port 
# ============================================================================
"""Commonly used data structures and functions."""

import enum
import torch
import torch.nn as nn
import numpy as np
# import ase
# from ase.neighborlist import neighbor_list

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def triangles_to_edges(faces, unique_op=True):
    """Computes mesh edges from triangles."""
    if faces.shape[-1] == 3:
    # collect edges from triangles
        edges = torch.cat([faces[:, 0:2],
                                         faces[:, 1:3],
                                         torch.stack([faces[:, 2], faces[:, 0]], axis=1)], dim=0)
    elif faces.shape[-1] == 2:
    # collect edges from a SINGLE edge rather than triangle
        edges = faces[:, 0:2]
    else:
        raise ValueError(f'ERROR triangles_to_edges expects 2 or 3 nodes per face')
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    if unique_op:
        receivers = torch.min(edges, dim=1)[0]
        senders = torch.max(edges, dim=1)[0]
        packed_edges = tf.bitcast(tf.stack([senders, receivers], dim=1), torch.int64)
        # remove duplicates and unpack
        unique_edges = tf.bitcast(tf.unique(packed_edges)[0], torch.int32)
        senders, receivers = tf.unstack(unique_edges, axis=1)
    else:
        receivers = edges[:,0]
        senders = edges[:,1]
    # create two-way connectivity
    return (torch.cat([senders, receivers], axis=0),
                    torch.cat([receivers, senders], axis=0))

def array2graph(frame, dim, periodic=True, device='cpu', time_dim=False):
    # frame shape: [batch, frame_shape, feature] without time dimension, otherwise [b, time, frame_shape, feature]
    graph = shape2graph(frame.shape[-dim-1:-1], frame.shape[0], periodic, device=device)
    if time_dim:
        graph['xtime'] = frame.permute((0,)+tuple(range(2,2+dim+1))+(1,)).reshape((-1,frame.shape[-1],frame.shape[1])).to(device)
        graph['x'] = frame.reshape((-1,frame.shape[-1]*frame.shape[1])).to(device)
    else:
        graph['x'] = frame.reshape((-1,frame.shape[-1])).to(device)
    graph['x_dense'] = graph['x']
    graph['x_idx'] = torch.arange(len(graph['x']))
    graph['x_fix'] = None
    return graph

import functools
@functools.lru_cache()
def shape2graph(shape, batch=1, periodic=True, device='cpu', aux=True):
    dim = len(shape)
    # print(f'debug frame {shape} {shape} {dim}')
    # n_node = np.prod(shape) * batch
    # mesh_pos = np.tile(np.stack(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij'), axis=-1),(batch,1)+(1,)*dim).reshape((-1, dim))
    mesh_pos_1 = np.stack(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij'), axis=-1).reshape((-1, dim))
    mesh_pos = np.tile(mesh_pos_1, (batch,1))
    # print(f'debug xyz {xyz} {mesh_pos}')
    if periodic:
        corner = mesh_pos_1
    else:
        corner = np.stack(np.meshgrid(*[np.arange(i-1) for i in shape], indexing='ij'), axis=-1).reshape((-1, dim))
    partitions = np.concatenate((np.eye(dim), -np.eye(dim)))
    partitions = np.stack([np.zeros_like(partitions), partitions], 1)[None,...]
    edge_index = (corner[:,None,None,:] + partitions) % shape
    edge_index = np.dot(edge_index, np.cumprod((1,)+shape[:0:-1])[::-1]).astype('int64').reshape(-1,2)
    edge_index = np.tile(edge_index[None,...], (batch,1,1)) + (np.arange(batch)*len(mesh_pos_1))[:,None,None]
    edge_index = edge_index.reshape(-1,2)

    graph = {}
    graph['mesh_pos'] = torch.tensor(mesh_pos, dtype=torch.float32, device=device)
    graph['node_type'] = torch.zeros((len(mesh_pos),1), dtype=torch.int64, device=device)
    graph['edge_index'] = torch.tensor(edge_index, dtype=torch.int64, device=device)
    edge_vec = graph['mesh_pos'][graph['edge_index'][:,1]]-graph['mesh_pos'][graph['edge_index'][:,0]]
    shape_pt = torch.tensor(shape, dtype=torch.float32, device=device)
    graph['edge_vec'] = edge_vec - (edge_vec/shape_pt).round()*shape_pt
    graph['edge_attr'] = torch.cat((graph['edge_vec'], torch.linalg.norm(graph['edge_vec'], dim=-1, keepdims=True)), -1)
    graph['lattice'] = torch.tensor(np.diag(shape), dtype=torch.float32, device=device)
    graph['inv_lattice'] = torch.tensor(np.diag(1/np.array(shape)), dtype=torch.float32, device=device)
    graph['batch_index'] = torch.arange(batch).repeat_interleave(len(mesh_pos_1))[:,None].to(device)
    if aux:
        graph['aux_vars'] = {'batch_cell_len': torch.tensor([batch]+list(shape)).float()[None,:].to(device),
      'batch': batch, 'shape':tuple(shape), 'n_dense': len(graph['mesh_pos']),
      'bwh':(batch,)+tuple(shape),
      'mesh_pos_dense':graph['mesh_pos'], 'node_type_dense':graph['node_type'],
      'batch_index_dense':graph['batch_index'], 'edge_index_dense':graph['edge_index'],
      'ijk2int': torch.tensor(np.cumprod([1]+list(shape)[::-1])[::-1].copy(), device=device, dtype=torch.int64)}
    # print(f'debug',graph['mesh_pos'].shape, graph['node_type'].shape, graph['edge_index'].shape, graph['lattice'].shape)
    return graph


def output_by_method(cur_x, x_update, method=None):
    if method is None:
        return x_update
    else:
        outs = []
        for i,m in enumerate(method):
            if m == 'id':
                out = x_update[...,i]
            elif m == 'fix':
                out = cur_x[...,i]
            elif m == 'res':
                out = x_update[...,i] + cur_x[...,i]
            elif m == 'sigmoid':
                out = torch.sigmoid(x_update[...,i])
            else:
                raise ValueError(f'unknown output channel method {m}')
            outs.append(out)
        return torch.stack(outs, -1)


def _get_neighbor_list(pos, cell_len, periodic, cutoff):
    tmp_cell = ase.Atoms(positions=pos.detach().cpu(), cell=cell_len, pbc=periodic)
    src, dst, edge_vec = neighbor_list("ijD", tmp_cell, cutoff, self_interaction=False)
    edge_ij = torch.from_numpy(np.array((src, dst))).to(pos.device)
    return edge_ij, torch.tensor(edge_vec, dtype=pos.dtype, device=pos.device)

@torch.jit.script
def _get_neighbor_list_nonpbc_bruteforce(p, cutoff: float):
    """Non-periodic. p shape=[N_atom, 3]"""
    vec = p[:,None,:] - p[None,:,:]
    dist = torch.linalg.norm(vec, dim=-1)
    ij = torch.nonzero(torch.logical_and(dist>0, dist<=cutoff))
    ij = ij[ij[:,0]!=ij[:,1]]
    vec = vec[(ij[:,0],ij[:,1])]
    return ij.T, vec

@torch.jit.script
def _get_neighbor_list_pbc_bruteforce(p, cutoff: float, lattice, inv_lattice):
    """lattice/inv_lattice should be 3x3 matrices"""
    vec = p[:,None,:] - p[None,:,:]
    vec = vec - torch.round(vec @ inv_lattice) @ lattice
    dist = torch.linalg.norm(vec, dim=-1)
    ij = torch.nonzero(torch.logical_and(dist>0, dist<=cutoff))
    ij = ij[ij[:,0]!=ij[:,1]]
    vec = vec[(ij[:,0],ij[:,1])]
    return ij.T, vec

@torch.jit.script
def _get_neighbor_list_pbc_bruteforce_orthorhombic(p, cutoff: float, lattice, inv_lattice):
    """lattice/inv_lattice should be 1x3 lattice constants (or 1/lattice)"""
    vec = p[:,None,:] - p[None,:,:]
    vec = vec - torch.round(vec * inv_lattice) * lattice
    dist = torch.linalg.norm(vec, dim=-1)
    ij = torch.nonzero(torch.logical_and(dist>0, dist<=cutoff))
    ij = ij[ij[:,0]!=ij[:,1]]
    vec = vec[(ij[:,0],ij[:,1])]
    return ij.T, vec


def get_neighbor_list(pos, cell_len, periodic, cutoff, batch=None, bruteforce=False, cell_inv=None):
    pos_list = torch.split(pos, tuple(batch[1:]-batch[:-1])) if batch is not None else (pos,)
    cell_list = cell_len if batch is not None else (cell_len,)
    cell_inv_list = cell_inv if batch is not None else (cell_inv,)
    # print(f'debug poslist {pos_list}')
    all_ij, all_vec = [], []
    # import time
    # tmp = torch.tensor([[9999],[1]], device=pos.device)
    for i, p in enumerate(pos_list):
        # t = time.time()
        if bruteforce and (not periodic):
            edge_ij, edge_vec = _get_neighbor_list_nonpbc_bruteforce(p, cutoff)
        elif bruteforce and periodic:
            edge_ij, edge_vec = _get_neighbor_list_pbc_bruteforce(p, cutoff, cell_list[i], cell_inv[i])
        else:
            edge_ij, edge_vec = _get_neighbor_list(p, cell_len, periodic, cutoff)
        # print(f'debug time {time.time()-t}')
        # t = time.time()
        # print(f'debug time {time.time()-t}')
        # print(f'debug diff', (torch.sort(edge_ij*tmp)[0] - torch.sort(Bedge_ij*tmp)[0]).abs().sum())
        # print(f'debug i {i} {p.shape} {p.device} {p.device} {p[:1]} ij {edge_ij.shape} ve {edge_vec.shape}')
        all_ij.append(edge_ij + batch[i] if batch is not None else edge_ij)
        all_vec.append(edge_vec)
    return torch.cat(all_ij, 1), torch.cat(all_vec)



def add_edge(edge_index, add_index, add_type, nnode, add_index_is_directed=True):
    """
    add_index: edge indices to be added.
    #### NOTE: assume add_index is directed, i.e. containing both ij and ji
    add_type: types of edge added
    """
    if not add_index_is_directed:
        add_index = torch.cat((add_index, torch.flip(add_index, (0,))), 1)
        add_type = add_type.view(-1,1).tile(2, 1)
    # print(add_index, edge_index, add_type)
    # return torch.nonzero(torch.zeros(Nmax, dtype=arr.dtype, device=arr.device).scatter_(0, arr, torch.ones_like(arr))).flatten()
    # TBD: loop over each graph of the batch
    flag_mat = torch.zeros((nnode, nnode), dtype=torch.float32, device=edge_index.device)#.scatter_(0, edge_index, torch.ones_like(edge_index[0]))
    # print(flag_mat, edge_index, edge_index.split(1))
    flag_mat[edge_index.split(1)] += 0.1
    # print(flag_mat, 'flag_mat[add_index.split(1)]', flag_mat[add_index.split(1)])
    flag_mat[add_index.split(1)] += add_type.view(1,-1)
    # print(flag_mat)
    # edge_index = torch.nonzero(flag_mat, as_tuple=True)
    # edge_type = flag_mat[edge_index].floor().int()
    # return torch.stack(edge_index), edge_type
    edge_index = torch.nonzero(torch.logical_and(flag_mat>0, flag_mat<1), as_tuple=True)
    edge_type = torch.zeros(edge_index[0].size(0), dtype=torch.long, device=edge_index[0].device)
    return torch.cat((add_index, torch.stack(edge_index)), 1), torch.cat((add_type, edge_type))


def g_get_edge_index(d, cutoff, fixedbond=True, periodic=False, edge_index_input=None):
    # print(f'pos {d.pos}')
    assert not periodic, NotImplementedError('periodic case to be put back')
    if edge_index_input is None:
        edge_index, edge_vec = get_neighbor_list(d.pos, [100.]*3, periodic, cutoff, bruteforce=True)
    else:
        edge_index = edge_index_input
    d['edge_index'] = edge_index
    print(f'debug edge index {edge_index.shape} vec ')
    # if self.mode in ('f', 'ef'):
    #     d.pos._requires_grad()
    if fixedbond:
        # print(f'debug fixed bond {g["fixedbond_index"]}')
        edge_index, edge_type = add_edge(edge_index, d["fixedbond_index"], d["fixedbond_type"], d.pos.size(0))
        # edge_type = nn.functional.one_hot(edge_type.long())
        d['edge_index'] = edge_index
        d['edge_type'] = edge_type
    d['edge_vec'] = d.pos[d['edge_index'][1]] - d.pos[d['edge_index'][0]]
    print(f'debug edge index {edge_index.shape} vec  ', d['edge_vec'].shape)


def batch_get_edge_index(g, cutoff, batch=None, fixedbond=True, periodic=False):
    if batch is None:
        g_get_edge_index(g, cutoff, fixedbond=fixedbond, periodic=periodic, edge_index_input=None)
    else:
        edge_index, edge_vec = get_neighbor_list(g.pos, [100.]*3, periodic, cutoff, batch=batch, bruteforce=True)
        g_get_edge_index(g, cutoff, fixedbond=fixedbond, periodic=periodic, edge_index_input=edge_index)


def preprocess_batch_graph(g, batch, reuse_edge=False, fixedbond=0, periodic=False, cutoff=4.5):
    if reuse_edge and hasattr(g, 'edge_index'):
        edge_index, edge_vec = g.edge_index, g.edge_vec
    else:
        edge_index, edge_vec = get_neighbor_list(g.pos, [100.]*3, periodic, cutoff, batch, bruteforce=True)
    if fixedbond:
        edge_index, edge_type = add_edge(edge_index, g["fixedbond_index"], g["fixedbond_type"], g.pos.size(0))
        edge_type = nn.functional.one_hot(edge_type.long())
        edge_vec = g.pos[edge_index[1]] - g.pos[edge_index[0]]
    else:
        edge_type = edge_vec.new_zeros((edge_vec.size(0), 0))
    g['edge_index'] = edge_index
    g['edge_vec'] = edge_vec
    g['edge_type'] = edge_type
    return g



def dataset_get_edge_index(dataset, args, fixedbond=True, pre_compute_edge_in_batch=False):
    debug = False
    if debug:
        import time
        t = time.time()
        print(f'debug start')
    # from torch_geometric.data import Data
    if pre_compute_edge_in_batch:
        edge_indices = [d.pos.shape[0] for d in dataset.flat]
        nnode = torch.tensor(edge_indices)
        for n in torch.unique(nnode):
            idx = torch.nonzero(nnode == n)[:,0]
            idx_batch = _get_neighbor_list_nonpbc_bruteforce_batch(torch.stack([dataset.flat[i].pos for i in idx]), args.edge_cutoff)[0]
            print('idx', idx.shape, len(idx_batch), idx_batch[0])
            for i,j in enumerate(idx):
                edge_indices[j] = idx_batch[i]
    for i, d in enumerate(dataset.flat):
        g_get_edge_index(d, args.edge_cutoff, fixedbond=fixedbond, periodic=args.periodic, 
          edge_index_input=(edge_indices[i] if pre_compute_edge_in_batch else None))
    if debug:
        print(f'debug done dataset edges', time.time()-t)
        exit()
    return dataset


def _get_neighbor_list_nonpbc_bruteforce_batch(p, cutoff, return_vec=False):
    dist = torch.linalg.norm(p[:,:,None,:] - p[:,None,:,:], dim=-1)
    ij = torch.nonzero(torch.logical_and(dist>0, dist<=cutoff))
    ij = ij[ij[:,1]!=ij[:,2]]
    # vec = p[ij[:,:2]] - p[ij[:,::2]]
    split_s = [ij[:,0]==i for i in range(p.shape[0])]
    return [ij[s][:,1:].T for s in split_s], "TBD" if return_vec else None

