import numpy as np
import torch
from matscipy.neighbours import neighbour_list as neighbor_list
import ase
# from ase.neighborlist import primitive_neighbor_list as neighbor_list ## TOO slow!
from torch import Tensor
from typing import Optional, Tuple, List

def atoms2pygdata(atoms_in, ase=True):
    from sklearn.preprocessing import LabelEncoder
    import torch
    from torch_geometric.data import Data
    atoms_list = atoms_in if isinstance(atoms_in, (list, tuple)) else [atoms_in]
    le = LabelEncoder()
    g_list = []
    for atoms in atoms_list:
        x = le.fit_transform(atoms.numbers)
        if ase:
            g= Data(
                x       = torch.tensor(x,               dtype=torch.long),
                pos     = torch.tensor(atoms.positions, dtype=torch.float),
                cell    = np.array(atoms.cell),
                cell_inv= np.linalg.inv(np.array(atoms.cell)) if np.all(atoms.pbc) else None,
                pbc     = atoms.pbc,
                numbers = atoms.numbers,
            )
        else:
            g= Data(
                x    = torch.tensor(x,                    dtype=torch.long),
                pos  = torch.tensor(atoms.positions,      dtype=torch.float),
                box  = torch.tensor(np.array(atoms.cell), dtype=torch.float).sum(dim=0),
            )
        g_list.append(g)
    return g_list if isinstance(atoms_in, (list, tuple)) else g_list[0]

def radius_graph(d, cutoff=4.0):
    i, j, D = neighbor_list(
        quantities="ijD",
        pbc=d.pbc,
        cell=d.cell,
        positions=d.pos.numpy(),
        cutoff=cutoff,
    )
    d.edge_index = torch.tensor(np.stack((i,j))).long()
    d.edge_features  = torch.tensor(D).float()
    return d

def periodic_radius_graph(x: Tensor, cutoff: float, cell: Tensor) -> Tuple[Tensor, Tensor]:
    i, j, S = neighbor_list(
        quantities="ijS",
        pbc=[True, True, True],
        cell=cell.numpy() if torch.is_tensor(cell) else cell,
        positions=x.numpy() if torch.is_tensor(x) else x,
        cutoff=cutoff,
    )
    edge_index = torch.tensor(np.stack((i,j))).long().to(x.device)
    S = torch.tensor(S, dtype=x.dtype, device=x.device)
    edge_vec = x[edge_index[1]]-x[edge_index[0]] + S @ cell
    return edge_index, edge_vec


def radius_graph(x, cutoff: float, cell, pbc, force_pos_differentiability=False) -> Tuple[Tensor, Tensor]:
    # print(f"debug radius_graph {x.shape=} {cell.shape=} {pbc=} {cutoff=} {x} {cell} {pbc}")
    # import ase, matscipy, time, matscipy.neighbours
    # for p in [[True,True,True],[False,False,False]]:
    #     for m in (matscipy.neighbours.neighbour_list, ase.neighborlist.neighbor_list):
    #         start_time = time.time()
    #         _,_,S = m("ijS",ase.Atoms(positions=x, cell=cell, pbc=pbc), 4.1)
    #         print(m, p, S.shape, time.time()-start_time)
    #     for m in (ase.neighborlist.primitive_neighbor_list,):
    #         start_time = time.time()
    #         m(quantities="ijS",
    #     pbc=pbc.numpy() if torch.is_tensor(pbc) else pbc,
    #     cell=cell.numpy() if torch.is_tensor(cell) else cell,
    #     positions=x.numpy() if torch.is_tensor(x) else x,
    #     cutoff=cutoff)
    #         print(m, p, S, time.time()-start_time)
    # breakpoint()
    # i, j, S = neighbor_list(
    #     quantities="ijS",
    #     pbc=pbc.numpy() if torch.is_tensor(pbc) else pbc,
    #     cell=cell.numpy() if torch.is_tensor(cell) else cell,
    #     positions=x.numpy() if torch.is_tensor(x) else x,
    #     cutoff=cutoff,
    # )
    i, j, D = neighbor_list("ijD", ase.Atoms(positions=x, cell=cell, pbc=pbc), cutoff=cutoff)
    edge_index = torch.tensor(np.stack((i,j))).long().to(x.device)
    # S = torch.tensor(S, dtype=x.dtype, device=x.device)
    # print(f"  debug {x.shape=} {x=} {cutoff=} {cell=} {pbc=} {S=} {edge_vec.shape=} {edge_vec[:5]=} {torch.is_tensor(cell)}" )
    # BUG in matscipy.neighbours here: S is not always reliable. It should be 0 for non-periodic, but turns out to be non-zero.
    # example, torch.tensor([[0.,0.,0.],[.9,0.,0.],[0.,.9,0.],[0.,0.,.9]]), 10.1, -torch.eye(3), [False,False,False]
    # And for mixed PBC like [True, True, False], S is also not reliable. So we cannot rely on S to determine the periodic image,
    # and we have to use the D vector.
    if force_pos_differentiability:
        edge_vec = x[edge_index[1]]-x[edge_index[0]]
        edge_vec = edge_vec + ((D-edge_vec) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec = torch.tensor(D, dtype=x.dtype, device=x.device)
    # if torch.is_tensor(cell): edge_vec += S @ cell
    return edge_index, edge_vec


def radius_graph_AB(x_A: Tensor, x_B: Tensor, cutoff: float, cell, pbc, force_pos_differentiability=False) -> Tuple[Tensor, Tensor]:
    """
    Build radius graph from points A to points B.

    Finds all neighbors within cutoff distance, where edges go from points in A to points in B.
    A and B can have different number of points.

    Args:
        x_A: Positions of points A, shape [N_A, 3]
        x_B: Positions of points B, shape [N_B, 3]
        cutoff: Cutoff radius for neighbor search
        cell: Unit cell for periodic boundary conditions
        pbc: Periodic boundary conditions [bool, bool, bool] or similar
        force_pos_differentiability: If True, compute edge_vec from positions to maintain gradients

    Returns:
        edge_index: Edge indices [2, num_edges], where edge_index[0] are indices in A (range 0 to N_A-1),
                    edge_index[1] are indices in B (range 0 to N_B-1)
        edge_vec: Edge vectors [num_edges, 3] pointing from A to B
    """
    N_A = x_A.shape[0]
    N_B = x_B.shape[0]

    # Concatenate A and B positions: [A_0, ..., A_{N_A-1}, B_0, ..., B_{N_B-1}]
    x_combined = torch.cat([x_A, x_B], dim=0)

    # Build neighbor list on combined positions
    i, j, D = neighbor_list("ijD", ase.Atoms(positions=x_combined, cell=cell, pbc=pbc), cutoff=cutoff)

    # Filter edges: keep only those from A (i < N_A) to B (j >= N_A)
    mask = (i < N_A) & (j >= N_A)
    i_filtered = i[mask]
    j_filtered = j[mask] - N_A  # Shift B indices to range [0, N_B-1]
    D_filtered = D[mask]

    # Convert to tensors
    edge_index = torch.tensor(np.stack((i_filtered, j_filtered))).long().to(x_A.device)

    # Compute edge vectors
    if force_pos_differentiability:
        edge_vec = x_B[edge_index[1]] - x_A[edge_index[0]]
        # Correct for periodic boundary conditions using the D vector
        edge_vec = edge_vec + ((torch.tensor(D_filtered, device=edge_vec.device) - edge_vec) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec = torch.tensor(D_filtered, dtype=x_A.dtype, device=x_A.device)

    return edge_index, edge_vec


def radius_graph_AB_BA(x_A: Tensor, x_B: Tensor, cutoff_AB: float, cutoff_BA: float, cell, pbc, force_pos_differentiability=False) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Build radius graphs from A to B and from B to A in a single call.

    More efficient than calling radius_graph_AB twice, as it only calls neighbor_list once
    with the maximum cutoff, then filters for both directions.

    Args:
        x_A: Positions of points A, shape [N_A, 3]
        x_B: Positions of points B, shape [N_B, 3]
        cutoff_AB: Cutoff radius for A -> B edges
        cutoff_BA: Cutoff radius for B -> A edges
        cell: Unit cell for periodic boundary conditions
        pbc: Periodic boundary conditions [bool, bool, bool] or similar
        force_pos_differentiability: If True, compute edge_vec from positions to maintain gradients

    Returns:
        ((edge_index_AB, edge_vec_AB), (edge_index_BA, edge_vec_BA)) where:
            - edge_index_AB: Edge indices [2, num_edges_AB] from A to B
            - edge_vec_AB: Edge vectors [num_edges_AB, 3] from A to B
            - edge_index_BA: Edge indices [2, num_edges_BA] from B to A
            - edge_vec_BA: Edge vectors [num_edges_BA, 3] from B to A
    """
    N_A = x_A.shape[0]
    N_B = x_B.shape[0]

    # Concatenate A and B positions: [A_0, ..., A_{N_A-1}, B_0, ..., B_{N_B-1}]
    x_combined = torch.cat([x_A, x_B], dim=0)

    # Build neighbor list with maximum cutoff
    cutoff_max = max(cutoff_AB, cutoff_BA)
    i, j, D = neighbor_list("ijD", ase.Atoms(positions=x_combined, cell=cell, pbc=pbc), cutoff=cutoff_max)

    # Compute distances for filtering
    D_norm = np.linalg.norm(D, axis=1)

    # Filter A -> B edges (i < N_A, j >= N_A, distance <= cutoff_AB)
    mask_AB = (i < N_A) & (j >= N_A) & (D_norm <= cutoff_AB)
    i_AB = i[mask_AB]
    j_AB = j[mask_AB] - N_A  # Shift B indices to range [0, N_B-1]
    D_AB = D[mask_AB]

    # Filter B -> A edges (i >= N_A, j < N_A, distance <= cutoff_BA)
    mask_BA = (i >= N_A) & (j < N_A) & (D_norm <= cutoff_BA)
    i_BA = i[mask_BA] - N_A  # Shift B indices to range [0, N_B-1]
    j_BA = j[mask_BA]
    D_BA = D[mask_BA]

    # Convert to tensors for A -> B
    edge_index_AB = torch.tensor(np.stack((i_AB, j_AB))).long().to(x_A.device)
    if force_pos_differentiability:
        edge_vec_AB = x_B[edge_index_AB[1]] - x_A[edge_index_AB[0]]
        edge_vec_AB = edge_vec_AB + ((torch.tensor(D_AB, device=edge_vec_AB.device) - edge_vec_AB) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec_AB = torch.tensor(D_AB, dtype=x_A.dtype, device=x_A.device)

    # Convert to tensors for B -> A
    edge_index_BA = torch.tensor(np.stack((i_BA, j_BA))).long().to(x_B.device)
    if force_pos_differentiability:
        edge_vec_BA = x_A[edge_index_BA[1]] - x_B[edge_index_BA[0]]
        edge_vec_BA = edge_vec_BA + ((torch.tensor(D_BA, device=edge_vec_BA.device) - edge_vec_BA) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec_BA = torch.tensor(D_BA, dtype=x_B.dtype, device=x_B.device)

    return (edge_index_AB, edge_vec_AB), (edge_index_BA, edge_vec_BA)


def radius_graph_AB_BA_BB(x_A: Tensor, x_B: Tensor, cutoff_AB: float, cutoff_BA: float, cutoff_BB: float, cell, pbc, force_pos_differentiability=False) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Build radius graphs from A to B, from B to A, and within B in a single call.

    More efficient than calling radius_graph_AB and radius_graph separately, as it only calls
    neighbor_list once with the maximum cutoff, then filters for all three edge types.

    Args:
        x_A: Positions of points A, shape [N_A, 3]
        x_B: Positions of points B, shape [N_B, 3]
        cutoff_AB: Cutoff radius for A -> B edges
        cutoff_BA: Cutoff radius for B -> A edges
        cutoff_BB: Cutoff radius for B -> B edges (within B)
        cell: Unit cell for periodic boundary conditions
        pbc: Periodic boundary conditions [bool, bool, bool] or similar
        force_pos_differentiability: If True, compute edge_vec from positions to maintain gradients

    Returns:
        ((edge_index_AB, edge_vec_AB), (edge_index_BA, edge_vec_BA), (edge_index_BB, edge_vec_BB)) where:
            - edge_index_AB: Edge indices [2, num_edges_AB] from A to B
            - edge_vec_AB: Edge vectors [num_edges_AB, 3] from A to B
            - edge_index_BA: Edge indices [2, num_edges_BA] from B to A
            - edge_vec_BA: Edge vectors [num_edges_BA, 3] from B to A
            - edge_index_BB: Edge indices [2, num_edges_BB] within B
            - edge_vec_BB: Edge vectors [num_edges_BB, 3] within B
    """
    N_A = x_A.shape[0]
    N_B = x_B.shape[0]

    # Concatenate A and B positions: [A_0, ..., A_{N_A-1}, B_0, ..., B_{N_B-1}]
    x_combined = torch.cat([x_A, x_B], dim=0)

    # Build neighbor list with maximum cutoff
    cutoff_max = max(cutoff_AB, cutoff_BA, cutoff_BB)
    i, j, D = neighbor_list("ijD", ase.Atoms(positions=x_combined, cell=cell, pbc=pbc), cutoff=cutoff_max)

    # Compute distances for filtering
    D_norm = np.linalg.norm(D, axis=1)

    # Filter A -> B edges (i < N_A, j >= N_A, distance <= cutoff_AB)
    mask_AB = (i < N_A) & (j >= N_A) & (D_norm <= cutoff_AB)
    i_AB = i[mask_AB]
    j_AB = j[mask_AB] - N_A  # Shift B indices to range [0, N_B-1]
    D_AB = D[mask_AB]

    # Filter B -> A edges (i >= N_A, j < N_A, distance <= cutoff_BA)
    mask_BA = (i >= N_A) & (j < N_A) & (D_norm <= cutoff_BA)
    i_BA = i[mask_BA] - N_A  # Shift B indices to range [0, N_B-1]
    j_BA = j[mask_BA]
    D_BA = D[mask_BA]

    # Filter B -> B edges (i >= N_A, j >= N_A, distance <= cutoff_BB)
    mask_BB = (i >= N_A) & (j >= N_A) & (D_norm <= cutoff_BB)
    i_BB = i[mask_BB] - N_A  # Shift B indices to range [0, N_B-1]
    j_BB = j[mask_BB] - N_A  # Shift B indices to range [0, N_B-1]
    D_BB = D[mask_BB]

    # Convert to tensors for A -> B
    edge_index_AB = torch.tensor(np.stack((i_AB, j_AB))).long().to(x_A.device)
    if force_pos_differentiability:
        edge_vec_AB = x_B[edge_index_AB[1]] - x_A[edge_index_AB[0]]
        edge_vec_AB = edge_vec_AB + ((torch.tensor(D_AB, device=edge_vec_AB.device) - edge_vec_AB) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec_AB = torch.tensor(D_AB, dtype=x_A.dtype, device=x_A.device)

    # Convert to tensors for B -> A
    edge_index_BA = torch.tensor(np.stack((i_BA, j_BA))).long().to(x_B.device)
    if force_pos_differentiability:
        edge_vec_BA = x_A[edge_index_BA[1]] - x_B[edge_index_BA[0]]
        edge_vec_BA = edge_vec_BA + ((torch.tensor(D_BA, device=edge_vec_BA.device) - edge_vec_BA) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec_BA = torch.tensor(D_BA, dtype=x_B.dtype, device=x_B.device)

    # Convert to tensors for B -> B
    edge_index_BB = torch.tensor(np.stack((i_BB, j_BB))).long().to(x_B.device)
    if force_pos_differentiability:
        edge_vec_BB = x_B[edge_index_BB[1]] - x_B[edge_index_BB[0]]
        edge_vec_BB = edge_vec_BB + ((torch.tensor(D_BB, device=edge_vec_BB.device) - edge_vec_BB) @ torch.linalg.inv(cell)).round().detach() @ cell
    else:
        edge_vec_BB = torch.tensor(D_BB, dtype=x_B.dtype, device=x_B.device)

    return (edge_index_AB, edge_vec_AB), (edge_index_BA, edge_vec_BA), (edge_index_BB, edge_vec_BB)


if __name__ == "__main__":
    print("="*60)
    print("Testing radius_graph_AB and radius_graph_AB_BA")
    print("="*60)

    # Test 1: Simple non-periodic case
    print("\n[Test 1] Non-periodic case")
    print("-"*60)

    # Create test data: 3 points in A, 4 points in B
    x_A = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    x_B = torch.tensor([
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [5.0, 5.0, 5.0],  # Far away point
    ])

    cell = torch.eye(3) * 10.0  # Large box, effectively non-periodic
    pbc = [False, False, False]
    cutoff_AB = 1.2
    cutoff_BA = 0.8

    print(f"Points A: {x_A.shape[0]}, Points B: {x_B.shape[0]}")
    print(f"Cutoff A->B: {cutoff_AB}, Cutoff B->A: {cutoff_BA}")

    # Test radius_graph_AB
    edge_index_AB, edge_vec_AB = radius_graph_AB(x_A, x_B, cutoff_AB, cell, pbc)
    print(f"\nA->B edges: {edge_index_AB.shape[1]}")
    print(f"  edge_index: {edge_index_AB.T.tolist()}")
    print(f"  edge distances: {torch.norm(edge_vec_AB, dim=1).tolist()}")

    edge_index_BA, edge_vec_BA = radius_graph_AB(x_B, x_A, cutoff_BA, cell, pbc)
    print(f"\nB->A edges: {edge_index_BA.shape[1]}")
    print(f"  edge_index: {edge_index_BA.T.tolist()}")
    print(f"  edge distances: {torch.norm(edge_vec_BA, dim=1).tolist()}")

    # Test radius_graph_AB_BA
    print("\n[Test 2] Using radius_graph_AB_BA (combined call)")
    print("-"*60)
    (edge_index_AB2, edge_vec_AB2), (edge_index_BA2, edge_vec_BA2) = \
        radius_graph_AB_BA(x_A, x_B, cutoff_AB, cutoff_BA, cell, pbc)

    print(f"A->B edges: {edge_index_AB2.shape[1]}")
    print(f"  edge_index: {edge_index_AB2.T.tolist()}")
    print(f"  edge distances: {torch.norm(edge_vec_AB2, dim=1).tolist()}")

    print(f"\nB->A edges: {edge_index_BA2.shape[1]}")
    print(f"  edge_index: {edge_index_BA2.T.tolist()}")
    print(f"  edge distances: {torch.norm(edge_vec_BA2, dim=1).tolist()}")

    # Verify consistency
    print("\n[Test 3] Consistency check")
    print("-"*60)

    # Sort edges for comparison
    def sort_edges(edge_index, edge_vec):
        # Sort by (i, j) tuple
        indices = torch.argsort(edge_index[0] * 1000 + edge_index[1])
        return edge_index[:, indices], edge_vec[indices]

    edge_index_AB_sorted, edge_vec_AB_sorted = sort_edges(edge_index_AB, edge_vec_AB)
    edge_index_AB2_sorted, edge_vec_AB2_sorted = sort_edges(edge_index_AB2, edge_vec_AB2)

    edge_index_BA_sorted, edge_vec_BA_sorted = sort_edges(edge_index_BA, edge_vec_BA)
    edge_index_BA2_sorted, edge_vec_BA2_sorted = sort_edges(edge_index_BA2, edge_vec_BA2)

    ab_index_match = torch.allclose(edge_index_AB_sorted.float(), edge_index_AB2_sorted.float())
    ab_vec_match = torch.allclose(edge_vec_AB_sorted, edge_vec_AB2_sorted, atol=1e-6)
    ba_index_match = torch.allclose(edge_index_BA_sorted.float(), edge_index_BA2_sorted.float())
    ba_vec_match = torch.allclose(edge_vec_BA_sorted, edge_vec_BA2_sorted, atol=1e-6)

    print(f"A->B edge_index match: {ab_index_match}")
    print(f"A->B edge_vec match: {ab_vec_match}")
    print(f"B->A edge_index match: {ba_index_match}")
    print(f"B->A edge_vec match: {ba_vec_match}")

    if ab_index_match and ab_vec_match and ba_index_match and ba_vec_match:
        print("✅ Consistency check PASSED!")
    else:
        print("❌ Consistency check FAILED!")

    # Test 4: Periodic boundary conditions
    print("\n[Test 4] Periodic boundary conditions")
    print("-"*60)

    x_A_pbc = torch.tensor([
        [0.1, 0.1, 0.1],
        [4.9, 4.9, 4.9],
    ])

    x_B_pbc = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
    ])

    cell_pbc = torch.eye(3) * 5.0
    pbc_true = [True, True, True]
    cutoff_pbc = 1.0

    print(f"Cell size: 5.0 x 5.0 x 5.0")
    print(f"Cutoff: {cutoff_pbc}")

    edge_index_pbc, edge_vec_pbc = radius_graph_AB(x_A_pbc, x_B_pbc, cutoff_pbc, cell_pbc, pbc_true)
    print(f"\nA->B edges with PBC: {edge_index_pbc.shape[1]}")
    print(f"  edge_index: {edge_index_pbc.T.tolist()}")
    print(f"  edge distances: {torch.norm(edge_vec_pbc, dim=1).tolist()}")
    print(f"  edge_vec sample: {edge_vec_pbc[:3].tolist()}")

    # Test that we find the periodic neighbor
    # Point A[1] at [4.9, 4.9, 4.9] should be within cutoff of B[0] at [0, 0, 0] due to PBC
    expected_neighbors = edge_index_pbc.shape[1]
    print(f"\nExpected to find neighbors across periodic boundary: {expected_neighbors > 0}")

    # Test 5: Different cutoffs in AB_BA
    print("\n[Test 5] Different cutoffs for A->B and B->A")
    print("-"*60)

    cutoff_AB_test = 1.5
    cutoff_BA_test = 0.6

    (edge_index_AB_diff, edge_vec_AB_diff), (edge_index_BA_diff, edge_vec_BA_diff) = \
        radius_graph_AB_BA(x_A, x_B, cutoff_AB_test, cutoff_BA_test, cell, pbc)

    print(f"With cutoff_AB={cutoff_AB_test}, cutoff_BA={cutoff_BA_test}:")
    print(f"  A->B edges: {edge_index_AB_diff.shape[1]}")
    print(f"  B->A edges: {edge_index_BA_diff.shape[1]}")
    print(f"  A->B distances: {torch.norm(edge_vec_AB_diff, dim=1).tolist()}")
    print(f"  B->A distances: {torch.norm(edge_vec_BA_diff, dim=1).tolist()}")

    # Verify all distances are within cutoffs
    ab_within_cutoff = torch.all(torch.norm(edge_vec_AB_diff, dim=1) <= cutoff_AB_test + 1e-6)
    ba_within_cutoff = torch.all(torch.norm(edge_vec_BA_diff, dim=1) <= cutoff_BA_test + 1e-6)

    print(f"\nAll A->B distances <= cutoff_AB: {ab_within_cutoff}")
    print(f"All B->A distances <= cutoff_BA: {ba_within_cutoff}")

    if ab_within_cutoff and ba_within_cutoff:
        print("✅ Cutoff check PASSED!")
    else:
        print("❌ Cutoff check FAILED!")

    # Test 6: Very different sizes (len(A)=1000, len(B)=10)
    print("\n[Test 6] Very different sizes: len(A)=1000, len(B)=10")
    print("-"*60)

    # Create random points
    torch.manual_seed(42)
    N_A_large = 1000
    N_B_small = 10

    x_A_large = torch.rand(N_A_large, 3) * 10.0
    x_B_small = torch.rand(N_B_small, 3) * 10.0

    cell_large = torch.eye(3) * 15.0
    pbc_large = [False, False, False]
    cutoff_large = 2.0

    print(f"Points A: {N_A_large}, Points B: {N_B_small}")
    print(f"Cutoff: {cutoff_large}")

    # Test radius_graph_AB
    edge_index_large, edge_vec_large = radius_graph_AB(x_A_large, x_B_small, cutoff_large, cell_large, pbc_large)

    print(f"\nA->B edges found: {edge_index_large.shape[1]}")

    # Check bounds for indices
    indices_A = edge_index_large[0]
    indices_B = edge_index_large[1]

    A_min, A_max = indices_A.min().item() if len(indices_A) > 0 else 0, indices_A.max().item() if len(indices_A) > 0 else 0
    B_min, B_max = indices_B.min().item() if len(indices_B) > 0 else 0, indices_B.max().item() if len(indices_B) > 0 else 0

    print(f"  Indices in A: min={A_min}, max={A_max} (expected range: [0, {N_A_large-1}])")
    print(f"  Indices in B: min={B_min}, max={B_max} (expected range: [0, {N_B_small-1}])")

    # Verify all indices are within bounds
    A_indices_valid = (indices_A >= 0).all() and (indices_A < N_A_large).all()
    B_indices_valid = (indices_B >= 0).all() and (indices_B < N_B_small).all()

    print(f"\nA indices within bounds [0, {N_A_large-1}]: {A_indices_valid}")
    print(f"B indices within bounds [0, {N_B_small-1}]: {B_indices_valid}")

    if A_indices_valid and B_indices_valid:
        print("✅ Index bounds check PASSED!")
    else:
        print("❌ Index bounds check FAILED!")

    # Also test with radius_graph_AB_BA
    print("\n[Test 6b] Same test with radius_graph_AB_BA")
    print("-"*60)

    (edge_index_AB_large, edge_vec_AB_large), (edge_index_BA_large, edge_vec_BA_large) = \
        radius_graph_AB_BA(x_A_large, x_B_small, cutoff_large, cutoff_large, cell_large, pbc_large)

    print(f"A->B edges: {edge_index_AB_large.shape[1]}, B->A edges: {edge_index_BA_large.shape[1]}")

    # Check A->B indices
    if edge_index_AB_large.shape[1] > 0:
        AB_A_valid = (edge_index_AB_large[0] >= 0).all() and (edge_index_AB_large[0] < N_A_large).all()
        AB_B_valid = (edge_index_AB_large[1] >= 0).all() and (edge_index_AB_large[1] < N_B_small).all()
        print(f"  A->B: A indices valid: {AB_A_valid}, B indices valid: {AB_B_valid}")
    else:
        AB_A_valid = AB_B_valid = True
        print(f"  A->B: No edges found")

    # Check B->A indices
    if edge_index_BA_large.shape[1] > 0:
        BA_B_valid = (edge_index_BA_large[0] >= 0).all() and (edge_index_BA_large[0] < N_B_small).all()
        BA_A_valid = (edge_index_BA_large[1] >= 0).all() and (edge_index_BA_large[1] < N_A_large).all()
        print(f"  B->A: B indices valid: {BA_B_valid}, A indices valid: {BA_A_valid}")
    else:
        BA_B_valid = BA_A_valid = True
        print(f"  B->A: No edges found")

    if AB_A_valid and AB_B_valid and BA_B_valid and BA_A_valid:
        print("✅ Index bounds check for AB_BA PASSED!")
    else:
        print("❌ Index bounds check for AB_BA FAILED!")

    # Test 7: radius_graph_AB_BA_BB with different sizes
    print("\n[Test 7] radius_graph_AB_BA_BB with very different sizes: len(A)=500, len(B)=20")
    print("-"*60)

    # Create random points with different sizes
    torch.manual_seed(123)
    N_A_test = 500
    N_B_test = 20

    x_A_test = torch.rand(N_A_test, 3) * 12.0
    x_B_test = torch.rand(N_B_test, 3) * 12.0

    cell_test = torch.eye(3) * 15.0
    pbc_test = [False, False, False]

    # Use different cutoffs for each edge type
    cutoff_AB_test = 1.8
    cutoff_BA_test = 1.5
    cutoff_BB_test = 3.2

    print(f"Points A: {N_A_test}, Points B: {N_B_test}")
    print(f"Cutoff A->B: {cutoff_AB_test}, Cutoff B->A: {cutoff_BA_test}, Cutoff B->B: {cutoff_BB_test}")

    # Call radius_graph_AB_BA_BB
    (edge_index_AB_test, edge_vec_AB_test), (edge_index_BA_test, edge_vec_BA_test), (edge_index_BB_test, edge_vec_BB_test) = \
        radius_graph_AB_BA_BB(x_A_test, x_B_test, cutoff_AB_test, cutoff_BA_test, cutoff_BB_test, cell_test, pbc_test)

    print(f"\nEdges found:")
    print(f"  A->B: {edge_index_AB_test.shape[1]}")
    print(f"  B->A: {edge_index_BA_test.shape[1]}")
    print(f"  B->B: {edge_index_BB_test.shape[1]}")

    # Check 1: Verify all distances are within respective cutoffs
    print(f"\n[Check 1] Distance bounds:")

    if edge_index_AB_test.shape[1] > 0:
        distances_AB = torch.norm(edge_vec_AB_test, dim=1)
        max_dist_AB = distances_AB.max().item()
        ab_within = (distances_AB <= cutoff_AB_test + 1e-6).all()
        print(f"  A->B: max_dist={max_dist_AB:.6f}, cutoff={cutoff_AB_test}, within={ab_within}")
    else:
        ab_within = True
        print(f"  A->B: No edges found")

    if edge_index_BA_test.shape[1] > 0:
        distances_BA = torch.norm(edge_vec_BA_test, dim=1)
        max_dist_BA = distances_BA.max().item()
        ba_within = (distances_BA <= cutoff_BA_test + 1e-6).all()
        print(f"  B->A: max_dist={max_dist_BA:.6f}, cutoff={cutoff_BA_test}, within={ba_within}")
    else:
        ba_within = True
        print(f"  B->A: No edges found")

    if edge_index_BB_test.shape[1] > 0:
        distances_BB = torch.norm(edge_vec_BB_test, dim=1)
        max_dist_BB = distances_BB.max().item()
        bb_within = (distances_BB <= cutoff_BB_test + 1e-6).all()
        print(f"  B->B: max_dist={max_dist_BB:.6f}, cutoff={cutoff_BB_test}, within={bb_within}")
    else:
        bb_within = True
        print(f"  B->B: No edges found")

    distances_check = ab_within and ba_within and bb_within
    if distances_check:
        print("✅ All distances within cutoffs!")
    else:
        print("❌ Some distances exceed cutoffs!")

    # Check 2: Verify index bounds
    print(f"\n[Check 2] Index bounds:")

    # Check A->B: source should be in [0, N_A_test-1], target in [0, N_B_test-1]
    if edge_index_AB_test.shape[1] > 0:
        AB_src_valid = (edge_index_AB_test[0] >= 0).all() and (edge_index_AB_test[0] < N_A_test).all()
        AB_dst_valid = (edge_index_AB_test[1] >= 0).all() and (edge_index_AB_test[1] < N_B_test).all()
        AB_src_range = f"[{edge_index_AB_test[0].min().item()}, {edge_index_AB_test[0].max().item()}]"
        AB_dst_range = f"[{edge_index_AB_test[1].min().item()}, {edge_index_AB_test[1].max().item()}]"
        print(f"  A->B: src in {AB_src_range}, expected [0, {N_A_test-1}]: {AB_src_valid}")
        print(f"        dst in {AB_dst_range}, expected [0, {N_B_test-1}]: {AB_dst_valid}")
        ab_indices_valid = AB_src_valid and AB_dst_valid
    else:
        ab_indices_valid = True
        print(f"  A->B: No edges to check")

    # Check B->A: source should be in [0, N_B_test-1], target in [0, N_A_test-1]
    if edge_index_BA_test.shape[1] > 0:
        BA_src_valid = (edge_index_BA_test[0] >= 0).all() and (edge_index_BA_test[0] < N_B_test).all()
        BA_dst_valid = (edge_index_BA_test[1] >= 0).all() and (edge_index_BA_test[1] < N_A_test).all()
        BA_src_range = f"[{edge_index_BA_test[0].min().item()}, {edge_index_BA_test[0].max().item()}]"
        BA_dst_range = f"[{edge_index_BA_test[1].min().item()}, {edge_index_BA_test[1].max().item()}]"
        print(f"  B->A: src in {BA_src_range}, expected [0, {N_B_test-1}]: {BA_src_valid}")
        print(f"        dst in {BA_dst_range}, expected [0, {N_A_test-1}]: {BA_dst_valid}")
        ba_indices_valid = BA_src_valid and BA_dst_valid
    else:
        ba_indices_valid = True
        print(f"  B->A: No edges to check")

    # Check B->B: both source and target should be in [0, N_B_test-1]
    if edge_index_BB_test.shape[1] > 0:
        BB_src_valid = (edge_index_BB_test[0] >= 0).all() and (edge_index_BB_test[0] < N_B_test).all()
        BB_dst_valid = (edge_index_BB_test[1] >= 0).all() and (edge_index_BB_test[1] < N_B_test).all()
        BB_src_range = f"[{edge_index_BB_test[0].min().item()}, {edge_index_BB_test[0].max().item()}]"
        BB_dst_range = f"[{edge_index_BB_test[1].min().item()}, {edge_index_BB_test[1].max().item()}]"
        print(f"  B->B: src in {BB_src_range}, expected [0, {N_B_test-1}]: {BB_src_valid}")
        print(f"        dst in {BB_dst_range}, expected [0, {N_B_test-1}]: {BB_dst_valid}")
        bb_indices_valid = BB_src_valid and BB_dst_valid
    else:
        bb_indices_valid = True
        print(f"  B->B: No edges to check")

    indices_check = ab_indices_valid and ba_indices_valid and bb_indices_valid
    if indices_check:
        print("✅ All indices within bounds!")
    else:
        print("❌ Some indices out of bounds!")

    # Check 3: Verify B->B edges are symmetric (if [i,j] exists, then [j,i] should also exist)
    print(f"\n[Check 3] B->B edge symmetry:")

    if edge_index_BB_test.shape[1] > 0:
        # Build set of all B->B edges
        bb_edges = set()
        for k in range(edge_index_BB_test.shape[1]):
            i, j = edge_index_BB_test[0, k].item(), edge_index_BB_test[1, k].item()
            bb_edges.add((i, j))

        # Check symmetry: for each (i, j), verify (j, i) exists
        symmetric = True
        missing_pairs = []
        for (i, j) in bb_edges:
            if (j, i) not in bb_edges:
                symmetric = False
                missing_pairs.append((j, i))

        if symmetric:
            print(f"  B->B edges are symmetric: ✅")
            print(f"    For each edge (i,j), the reverse edge (j,i) exists")
            symmetry_check = True
        else:
            print(f"  B->B edges are NOT symmetric: ❌")
            print(f"    Missing reverse edges: {missing_pairs[:5]}{'...' if len(missing_pairs) > 5 else ''}")
            symmetry_check = False
    else:
        print(f"  B->B: No edges to check (trivially symmetric)")
        symmetry_check = True

    # Overall test 7 result
    if distances_check and indices_check and symmetry_check:
        print("\n✅ Test 7 (radius_graph_AB_BA_BB) PASSED!")
    else:
        print("\n❌ Test 7 (radius_graph_AB_BA_BB) FAILED!")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)