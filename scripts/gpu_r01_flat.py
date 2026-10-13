"""F62 contiguous storage, preserving per-parameter norm and AdamW operations."""
import torch


class FlatMasterChildSync:
    def __init__(self, master, child, optimizer):
        masters = dict(master.named_parameters())
        self.pairs = [(masters[n], p) for n, p in child.named_parameters()]
        self.groups = []
        self.gradient_views = {}
        with torch.no_grad():
            for dtype in dict.fromkeys(c.dtype for _, c in self.pairs):
                pairs = [(m, c) for m, c in self.pairs if c.dtype == dtype]
                # CUDA allocations are 256-byte aligned. Preserve that contract for
                # every weight: cuBLAS may choose a different reduction for unaligned slices.
                slots = [((m.numel() + 127) // 128) * 128 for m, _ in pairs]
                source = torch.zeros(sum(slots), device=pairs[0][0].device)
                target = torch.zeros(sum(slots), device=pairs[0][1].device, dtype=dtype)
                gradients = torch.empty_like(source)
                offset = 0
                for (m, c), slot in zip(pairs, slots):
                    size = m.numel()
                    source[offset:offset + size].copy_(m.detach().flatten())
                    target[offset:offset + size].copy_(c.detach().flatten())
                    m.data = source[offset:offset + size].view_as(m)
                    c.data = target[offset:offset + size].view_as(c)
                    self.gradient_views[id(m)] = gradients[offset:offset + size].view_as(m)
                    offset += slot
                self.groups.append((source, target))
            # Retain parameter group/order and lazy step semantics. Only moment storage changes.
            self.moment_buffers = []
            for group in optimizer.param_groups:
                parameters = group['params']
                if not parameters:
                    continue
                for key in ('exp_avg', 'exp_avg_sq'):
                    buffer = torch.zeros(sum(p.numel() for p in parameters), device=parameters[0].device)
                    self.moment_buffers.append(buffer)
                    offset = 0
                    for p in parameters:
                        state = optimizer.state[p]
                        state.setdefault('step', torch.tensor(0.0))
                        state[key] = buffer[offset:offset + p.numel()].view_as(p)
                        offset += p.numel()

    @torch.no_grad()
    def downcast(self):
        for source, target in self.groups:
            target.copy_(source)

    @torch.no_grad()
    def transfer_grads(self):
        groups = {}
        for master, child in self.pairs:
            if master.grad is not None:
                raise RuntimeError('F62 requires cleared master gradients')
            if child.grad is not None:
                master.grad = self.gradient_views[id(master)]
                targets, sources = groups.setdefault(child.grad.dtype, ([], []))
                targets.append(master.grad)
                sources.append(child.grad)
        for targets, sources in groups.values():
            torch._foreach_copy_(targets, sources)
            torch._foreach_add_(targets, 0.0)
            torch._foreach_zero_(sources)
