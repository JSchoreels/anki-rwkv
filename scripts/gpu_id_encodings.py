"""F19: batched CPU ID draws preserving original set order and RNG stream."""
import numpy as np
import torch
from rwkv.config import ID_SPLIT


def batched_id_table(ids, dimensions, dtype):
    if ids.device.type != "cpu" or ids.ndim != 1 or dimensions < 1:
        raise ValueError("Batched ID encoding requires one-dimensional CPU IDs")
    identifiers = list(set(ids.tolist()))
    unique, inverse = np.unique(ids.numpy(), return_inverse=True)
    # Calendar draws remain after all four modules, in their original order.
    draws = torch.randint(0, ID_SPLIT, (len(identifiers), dimensions), device="cpu").to(dtype)
    draws -= (ID_SPLIT - 1) / 2
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    order = torch.tensor([positions[identifier] for identifier in unique], dtype=torch.long)
    return draws.index_select(0, order), torch.from_numpy(inverse)
