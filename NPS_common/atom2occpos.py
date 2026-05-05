__author__ = 'Fei Zhou'

import numpy as np
import torch

from NPS_common.scatter import scatter_sum


def atom2occpos(lattvec, positions, shape, periodic=True,
                    embed=None, dim=3, **kwx):
    """
    Each atom is assigned to a voxel, with occupancy defined by embed
    """
    device = positions.device
    dtype = positions.dtype

    nvoxel = torch.prod(shape)
    shape = shape.to(device)
    # now assume orthorhombic supercell
    box = torch.diag(lattvec).to(positions.dtype)/shape
    lattice = lattvec.to(positions.dtype)
    lattice = lattice.to(device).diag()
    # global nbs_all, nbs_222
    # nbs_all = nbs_all.to(positions.device)
    # nbs_222 = nbs_222.to(positions.device)

    x = positions.reshape(-1, dim)
    if periodic == True:
        x = x - torch.floor(x /lattice) * lattice
        i_a = ((x.detach()/box)% shape).floor()
        i_a_idx = (i_a%shape).long()
        frac = (x/box)% shape - i_a_idx
    else:
        i_a = ((x.detach()/box)).floor()
        i_a_idx = (i_a).long()
        in_bounds = ((i_a_idx >= 0) & (i_a_idx <= shape-1)).all(dim=-1)
        i_a_idx = i_a_idx[in_bounds]
        i_a = i_a[in_bounds]
        x = x[in_bounds]
        embed = embed[in_bounds]
        frac = (x/box) - i_a_idx

    # i_a_idx = (i_a_idx.reshape(-1,3).double() @ cell_typ2int[:3].double()).long()
    i_a_idx_1d = i_a_idx[:,0]*shape[1]*shape[2] + i_a_idx[:,1]*shape[2] + i_a_idx[:,2]

    arr = scatter_sum(torch.cat((embed, frac), -1), i_a_idx_1d, dim=0, dim_size=nvoxel)
    return arr.reshape(shape.tolist()+[-1])

