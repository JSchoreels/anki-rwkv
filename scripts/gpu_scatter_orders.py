"""F53: immutable CPU scatter orders, shipped in the existing packed storage."""
import numpy as np
import torch


def stable_orders(index, rows):
    values = index.numpy().reshape(-1)
    order = np.argsort(values, kind="stable").astype(np.int64, copy=False)
    starts = np.searchsorted(values[order], np.arange(rows + 1, dtype=np.int64)).astype(np.int64, copy=False)
    return torch.from_numpy(order), torch.from_numpy(starts)


def prepare_orders(batch):
    rows = batch.start.shape[0]
    for i, (gathers, shifts, lengths) in enumerate(zip(batch.sub_gather, batch.time_shift_selects, batch.sub_gather_lens)):
        for j, (gather, shift, length) in enumerate(zip(gathers, shifts, lengths)):
            prefix = f"scatter_{i}_{j}_"
            row_order, row_starts = stable_orders(gather, rows)
            keys = (np.arange(shift.numel() // length, dtype=np.int64)[:, None] * length
                    + shift.numpy().reshape(-1, length)).reshape(-1)
            shift_order, shift_starts = stable_orders(torch.from_numpy(keys), len(keys))
            for name, value in (("row_order", row_order), ("row_starts", row_starts),
                                ("shift_order", shift_order), ("shift_starts", shift_starts)):
                batch.packed[prefix + name] = value
        rows = sum(gather.numel() for gather in gathers)
    return batch


def install_orders(batch):
    from rwkv.model import exact_index
    exact_index._orders.clear()
    exact_index._row_orders.clear()
    metadata = getattr(batch, "scatter_orders", None)
    if metadata is None:
        raise ValueError("F53 requires worker-prepared packed scatter metadata")
    for i, (gathers, shifts, lengths) in enumerate(zip(batch.sub_gather, batch.time_shift_selects, batch.sub_gather_lens)):
        for j, (gather, shift, length) in enumerate(zip(gathers, shifts, lengths)):
            prefix = f"scatter_{i}_{j}_"
            select = shift.view(-1, length)
            key = (select.data_ptr(), *select.shape, select.stride(), select.dtype, select._version, select.device)
            exact_index._orders[key] = (select, metadata[prefix + "shift_order"], metadata[prefix + "shift_starts"])
            row_key = (gather.data_ptr(), gather.numel(), gather._version, gather.device)
            exact_index._row_orders[row_key] = (gather, metadata[prefix + "row_order"], metadata[prefix + "row_starts"])
