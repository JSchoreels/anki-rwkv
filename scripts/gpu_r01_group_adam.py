"""F62 whole active AdamW groups, bucketed by per-parameter step and dtype."""
import torch


def install_group_adam(optimizer, reusable=False):
    """Keep the public parameter/state groups; pack only active equal-step buckets."""
    cache = {}

    @torch.no_grad()
    def step(closure=None):
        if closure is not None:
            raise ValueError('R01 grouped Adam does not use closures')
        from torch.optim._functional import adamw
        for group in optimizer.param_groups:
            buckets = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                if not state:
                    state['step'] = torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                key = (p.device, p.dtype, float(state['step']))
                buckets.setdefault(key, []).append(p)
            for params in buckets.values():
                states = [optimizer.state[p] for p in params]
                sources = [[p.view(-1) for p in params], [p.grad.view(-1) for p in params],
                           [s['exp_avg'].view(-1) for s in states], [s['exp_avg_sq'].view(-1) for s in states]]
                key = tuple(id(p) for p in params)
                if reusable:
                    buffers = cache.get(key)
                    if buffers is None:
                        buffers = cache[key] = [torch.empty(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype) for _ in sources]
                    for values, target in zip(sources, buffers):
                        torch.cat(values, out=target)
                else:
                    buffers = [torch.cat(values) for values in sources]
                packed, grad, avg, sq = buffers
                counter = states[0]['step'].clone()
                adamw([packed], [grad], [avg], [sq], [], [counter], foreach=True,
                      capturable=False, differentiable=False, fused=False, amsgrad=False,
                      beta1=group['betas'][0], beta2=group['betas'][1], lr=group['lr'],
                      weight_decay=group['weight_decay'], eps=group['eps'], maximize=group['maximize'])
                sizes = [p.numel() for p in params]
                for values, buffer in zip((sources[0], sources[2], sources[3]), (packed, avg, sq)):
                    torch._foreach_copy_(values, list(buffer.split(sizes)))
                for state in states:
                    state['step'].copy_(counter)
    optimizer.step = step
