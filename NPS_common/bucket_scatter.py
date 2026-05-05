import torch

def bucket_scatter(a: torch.Tensor, index: torch.Tensor, N: int, k: int):
    """
    Ultra-efficient vectorized implementation with minimal loops.
    Uses advanced PyTorch operations for maximum performance.
    """
    M, dim = a.shape
    assert index.shape == (M,), "index must be shape (M,)"
    assert index.dtype == torch.long, "index must be LongTensor"
    device = a.device

    # Count occurrences using bincount (fastest counting method)
    counts = torch.bincount(index, minlength=N)
    max_count = counts.max().item()

    if max_count > k:
        print(f"Error: Some bucket had {max_count} entries, but k={k}. data:")
        print(a[torch.argmax(counts)==index])
        return
        # import numpy as np
        # np.savetxt("debug_index.txt", index.cpu().numpy())
        # np.savetxt("debug_pos.txt", a[:,:3].cpu().numpy())
        # import numpy as np
        # import ase, ase.io
        # ase.io.write("debug.extxyz", ase.Atoms(positions=a[:,:3].numpy(), cell=cell))
        raise ValueError(f"Some bucket had {max_count} entries, but k={k}. "
                        f"Increase k or pre-trim input.")

    # Create a mapping from linear index to (bucket, slot) coordinates
    # This uses the fact that we can compute cumulative positions
    range_M = torch.arange(M, device=device)

    # Sort by index to group elements
    sort_perm = torch.argsort(index)
    sorted_index = index[sort_perm]

    # Create slot numbers using cumulative operations within groups
    # Use the trick: cumcount = arange(M) - cumulative_index_starts[index]
    unique_sorted, inverse = torch.unique_consecutive(sorted_index, return_inverse=True)

    # Find where each new group starts
    group_starts = torch.zeros(len(unique_sorted) + 1, dtype=torch.long, device=device)
    group_starts[1:] = torch.cumsum(torch.bincount(inverse), dim=0)

    # Compute slot for each element in sorted order
    slots_sorted = range_M - group_starts[inverse]

    # Unsort to get back to original order
    slots = torch.zeros(M, dtype=torch.long, device=device)
    slots[sort_perm] = slots_sorted

    # Scatter into output tensor
    out = torch.zeros(N, k, dim, device=device, dtype=a.dtype)
    out[index, slots] = a
    return out


def bucket_scatter_no_sort(a: torch.Tensor, index: torch.Tensor, N: int, k: int):
    """
    Scatter rows of `a` into buckets by `index`, placing them into slots along a new axis.

    Args:
        a:     Tensor of shape (M, dim)
        index: Long tensor of shape (M,), values in [0, N-1]
        N:     Number of buckets
        k:     Max number of entries per bucket

    Returns:
        out: Tensor of shape (Nk, dim) where
             out[index[i], slot[i], :] = a[i, :]
             and slot[i] is the position of row i within its bucket.
             Unused slots are zeros.
    """
    M, dim = a.shape
    assert index.shape == (M,), "index must be shape (M,)"
    assert index.dtype == torch.long, "index must be LongTensor"

    device = a.device

    counts = torch.bincount(index, minlength=N)
    assert k >= counts.max(), ValueError(f" Too small{k=} but {counts.max()=}")

    # --- Step 1: count occurrences per index
    counts = torch.zeros(N, dtype=torch.long, device=device)
    slots = torch.empty(M, dtype=torch.long, device=device)

    for i in range(M):
        j = index[i].item()
        slots[i] = counts[j]
        counts[j] += 1

    # --- Safety check
    if (slots >= k).any():
        raise ValueError(f"Some bucket had more than {k} entries. "
                         f"Increase k or pre-trim input.")

    # --- Step 2: scatter into output
    out = torch.zeros(N, k, dim, device=device, dtype=a.dtype)
    out[index, slots] = a
    return out


if __name__ == "__main__":
    M, N, k, dim = 12, 9, 2, 3
    torch.manual_seed(0)

    a = torch.arange(M*dim).reshape(M, dim).float()
    index = torch.randint(0, N, (M,))

    out = bucket_scatter(a, index, N, k)

    print("index:", index)
    print("out shape:", out.shape)   # (N, k, dim)
    print(out)
