"""F23/F20 update helpers: fewer launches with values identical to the accepted path.

F23 groups the FP32-master/child parameter traffic into foreach launches. F20
pins and copies a packed worker buffer once and rebuilds its typed views on the
device. Both preserve every value; only launch count and host work change.
"""

import torch


class MasterChildSync:
    """Grouped replacement for copy_downcast_ and transfer_child_grad_to_master."""

    def __init__(self, master, child):
        masters = dict(master.named_parameters())
        self.pairs = [(masters[name], parameter) for name, parameter in child.named_parameters()]
        self.downcast_groups = {}
        for name, parameter in child.named_parameters():
            # The child is already selectively cast; its dtype is the copy_downcast_ target.
            sources, targets = self.downcast_groups.setdefault(parameter.dtype, ([], []))
            sources.append(masters[name].data)
            targets.append(parameter.data)

    @torch.no_grad()
    def downcast(self):
        # copy_ rounds once to the target dtype, exactly as master.to(dtype) then copy_.
        for sources, targets in self.downcast_groups.values():
            torch._foreach_copy_(targets, sources)

    @torch.no_grad()
    def transfer_grads(self):
        if any(master.grad is not None for master, _ in self.pairs):
            raise RuntimeError("Grouped gradient transfer expects cleared master gradients")
        groups = {}
        for master, child in self.pairs:
            # Parameters without a child gradient keep master.grad None, so AdamW skips them as before.
            if child.grad is not None:
                master.grad = torch.empty_like(master)
                targets, sources = groups.setdefault(child.grad.dtype, ([], []))
                targets.append(master.grad)
                sources.append(child.grad)
        for targets, sources in groups.values():
            torch._foreach_copy_(targets, sources)
            # The original zeros_like(...).add_(g) maps -0.0 to +0.0; adding +0.0 preserves that bit pattern.
            torch._foreach_add_(targets, 0.0)
            torch._foreach_zero_(sources)


def bulk_pin(batch, map_batch, tensors):
    """Pin each shared packed storage once and keep views addressing the pinned copy."""
    storages = {}
    for value in tensors(batch):
        storage = value.untyped_storage()
        if storage.data_ptr() not in storages:
            whole = torch.empty(0, dtype=torch.uint8).set_(storage)
            storages[storage.data_ptr()] = whole.pin_memory()

    def pinned_view(value):
        whole = storages[value.untyped_storage().data_ptr()]
        start = value.storage_offset() * value.element_size()
        if not value.is_contiguous():
            raise ValueError("Bulk staging requires contiguous packed views")
        return whole[start:start + value.numel() * value.element_size()].view(value.dtype).view(value.shape)

    return map_batch(batch, pinned_view), storages


def bulk_to_device(batch, map_batch, storages, device):
    """One asynchronous copy per pinned storage; typed views are rebuilt on the device."""
    moved = {pointer: whole.to(device, non_blocking=True) for pointer, whole in storages.items()}
    pinned = {pointer: whole.data_ptr() for pointer, whole in storages.items()}

    def device_view(value):
        pointer = next(key for key, base in pinned.items()
                       if base <= value.data_ptr() < base + storages[key].numel())
        start = value.data_ptr() - pinned[pointer]
        return moved[pointer][start:start + value.numel() * value.element_size()].view(value.dtype).view(value.shape)

    return map_batch(batch, device_view)
