import torch
from torch_geometric.transforms import BaseTransform

class RadiusGraph_brute(BaseTransform):
    """
    Only works for a single data object, not batches of data.
    Args:
        cutoff (float): Cutoff distance within which pairs of nodes would
            be considered connected.
    """
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self, data, cutoff=None, edge_attr='D'):
        p, lattice, pbc, numbers = data.pos, data.lattice, data.pbc, data.numbers
        inv_lattice = data.inv_lattice
        cutoff = cutoff or self.cutoff
        assert pbc[0]==pbc[1] and pbc[0]==pbc[2]
        pbc = pbc[0]

        vec = p[:,None,:] - p[None,:,:]
        # print(p, lattice)
        if pbc:
            vec = vec - torch.round(vec @ inv_lattice) @ lattice
        dist = torch.linalg.norm(vec, dim=-1)
        dist.fill_diagonal_(0.)
        ij = torch.nonzero(torch.logical_and(dist>0., dist<=cutoff)).T
        if hasattr(data, "strong_bnd_index"):
            strong_bnd_index = data.strong_bnd_index
            num_nodes = p.size(0)
            # ij = torch.nonzero(torch.logical_or(torch.logical_and(dist>0, dist<=cutoff), M))
            M = torch.zeros((num_nodes, num_nodes), dtype=torch.uint8, device=p.device)
            M[tuple(ij)] = 1
            M[tuple(strong_bnd_index)] = 2
            # NOTE: edge_index_to_mask might not be a subset of edge_index. Expand edge_index to make sure!!
            data.edge_index = torch.nonzero(M).T
            data.edge_mask = (M[tuple(data.edge_index)] > 1).long()
        else:
            data.edge_index = ij
        edge_index = tuple(data.edge_index)
        edge_vec = vec[edge_index]
        edge_len = dist[edge_index][:,None]
        if edge_attr == 'D':
            data.edge_attr = edge_vec
        elif edge_attr == 'd':
            data.edge_attr = edge_len
        elif edge_attr == 'dD':
            data.edge_attr = torch.hstack((edge_len, edge_vec))
        elif edge_attr == 'Dd':
            data.edge_attr = torch.hstack((edge_vec, edge_len))
        else:
            raise ValueError(f"Unknown {edge_attr=} requested")
        return data

