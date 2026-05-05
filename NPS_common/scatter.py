# from mace/tools/scatter.py
"""basic scatter_sum operations from torch_scatter from
https://github.com/mir-group/pytorch_runstats/blob/main/torch_runstats/scatter_sum.py
Using code from https://github.com/rusty1s/pytorch_scatter, but cut down to avoid a dependency.
PyTorch plans to move these features into the main repo, but until then,
to make installation simpler, we need this pure python set of wrappers
that don't require installing PyTorch C++ extensions.
See https://github.com/pytorch/pytorch/issues/63780.
"""

from typing import Optional, Tuple

import torch


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_reduce_(dim, index, src, reduce=reduce)
    else:
        return out.scatter_reduce_(dim, index, src, reduce=reduce)


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for torch_scatter.scatter_max.

    Returns (values, argmax) matching torch_scatter's API exactly:
        values  — the per-index maximum values  (same shape as out would be)
        argmax  — the index of the max element in src for each output slot

    Unfilled slots (indices that received no src element) are:
        values  → -inf  (matches torch_scatter behaviour)
        argmax  → 0     (matches torch_scatter behaviour)

    Notes
    -----
    torch_scatter.scatter_max(src, index, dim, out, dim_size) returns a
    *named tuple* with fields .values and .argmax.  We return a plain tuple
    (values, argmax) which unpacks identically:

        max_vals, argmax = scatter_max(src, index, dim=0, dim_size=N)
        # or
        result = scatter_max(src, index, dim=0, dim_size=N)
        max_vals = result[0]   # same as result.values in torch_scatter
    """
    if dim < 0:
        dim = src.dim() + dim

    # resolve output size along `dim`
    if dim_size is None:
        if out is not None:
            dim_size = out.size(dim)
        elif index.numel() == 0:
            dim_size = 0
        else:
            dim_size = int(index.max().item()) + 1

    # broadcast index to match src shape
    index_bc = _broadcast(index, src, dim)

    # output shape
    out_size = list(src.size())
    out_size[dim] = dim_size

    # initialise values to -inf so any real value beats the default
    if out is not None:
        values = out
    else:
        values = torch.full(out_size, float('-inf'),
                            dtype=src.dtype, device=src.device)

    argmax = torch.zeros(out_size, dtype=torch.long, device=src.device)

    # scatter_reduce with 'amax' fills values; then a second pass gets argmax
    values.scatter_reduce_(dim, index_bc, src, reduce="amax", include_self=True)

    # argmax: for each output slot, find which src position achieved the max
    # Strategy: mask src positions that equal the max at their target slot,
    # then take the last (or first) such position via scatter.
    # We use "last writer wins" via a linear position tensor.
    max_at_target = values.gather(dim, index_bc)       # [same shape as src]
    is_max = (src == max_at_target)                    # bool mask

    # position indices along the scatter dimension
    pos_size  = [1] * src.dim()
    pos_size[dim] = src.size(dim)
    positions = torch.arange(src.size(dim), device=src.device).view(pos_size)
    positions = positions.expand_as(src)

    # scatter the position of the winning element (last tie wins, consistent
    # with torch_scatter which returns the last occurrence)
    argmax.scatter_reduce_(
        dim,
        index_bc,
        torch.where(is_max, positions, torch.zeros_like(positions)),
        reduce="amax",
        include_self=True,
    )

    return values, argmax


def scatter_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = _broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out