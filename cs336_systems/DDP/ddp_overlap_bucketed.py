import torch
import torch.distributed as dist

class Bucket:

    def __init__(self, params):
        self.params = params
        self.param_to_idx = {}
        self.offsets = []
        idx = 0
        for i, p in enumerate(self.params):
            self.offsets.append(idx)
            self.param_to_idx[p] = i
            idx += p.numel()
        self.buffer = torch.zeros(idx, dtype=params[0].dtype, device=params[0].device)
        self._ready = 0
        self.handle = None


    def ready(self, param):
        idx = self.param_to_idx[param]
        start = self.offsets[idx]
        end = start + self.params[idx].numel()
        self.buffer[start:end].copy_(param.grad.view(-1))
        self._ready += 1
        if self._ready == len(self.params):
            self.handle = dist.all_reduce(self.buffer, async_op=True)


    def wait(self):
        if self.handle is None:
            self._ready = 0
            return
        self.handle.wait()
        self.buffer /= dist.get_world_size()
        offset = 0
        for p in self.params:
            start, end = offset, offset + p.numel()
            p.grad.copy_(self.buffer[start:end].view(p.shape))
            offset = end
        self.handle = None
        self._ready = 0


class DDPModuleWrapper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        self.module = module
        param_size = 0
        bucket_params = []
        self.param_to_bucket = {}
        self.buckets = []
        bucket_size_b = bucket_size_mb * 1024 * 1024

        params = list(module.parameters())
        for p in params[::-1]:
            dist.broadcast(p.data, src=0)

            if not p.requires_grad:
                continue

            p_size = p.numel() * p.element_size()
            if param_size + p_size > bucket_size_b and bucket_params:
                bucket = Bucket(bucket_params)
                self.buckets.append(bucket)
                for param in bucket_params:
                    self.param_to_bucket[param] = bucket
                param_size = 0
                bucket_params = []

            param_size += p_size
            bucket_params.append(p)
        
        if bucket_params:
            bucket = Bucket(bucket_params)
            self.buckets.append(bucket)
            for p in bucket_params:
                self.param_to_bucket[p] = bucket

        for p in params:
            if not p.requires_grad:
                continue
            bucket = self.param_to_bucket[p]
            p.register_post_accumulate_grad_hook(lambda p, b=bucket: b.ready(p))


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            bucket.wait()

