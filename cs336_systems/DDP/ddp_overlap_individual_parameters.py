import torch
import torch.distributed as dist


class DDPModuleWrapper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self.module = module
        self._futures = []
        
        # Rank 0 broadcast model parameters to other workers
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
            if not p.requires_grad:
                continue
            p.register_post_accumulate_grad_hook(self._grad_hook)


    def _grad_hook(self, p: torch.Tensor):
        if p.grad is None:
            return
        handle = dist.all_reduce(p.grad, async_op=True)
        self._futures.append((handle, p.grad))


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle, grad in self._futures:
            handle.wait()
            grad /= world_size
        self._futures.clear()