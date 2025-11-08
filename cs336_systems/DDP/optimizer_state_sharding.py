import torch
import torch.distributed as dist
from typing import Any, Type, Callable


class DDPOptimizerWrapper(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.initializing = True
        param_list = list(params)
        super().__init__(param_list, kwargs)
        self.initializing = False
        self.param_to_rank = {}

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        n = len(param_list)
        m = (n - 1) // world_size + 1

        for i, p in enumerate(param_list):
            self.param_to_rank[p] = i // m

        sharded_params = param_list[rank * m : (rank + 1) * m]
        self.optimizer = optimizer_cls(sharded_params, **kwargs)


    def step(self, closure: Callable|None = None, **kwargs):
        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                dist.broadcast(p.data, src=self.param_to_rank[p])
    
        return loss


    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)

        if self.initializing:
            return

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        params = param_group["params"]
        n = len(params)
        m = (n - 1) // world_size + 1
        
        for i, p in enumerate(params):
            self.param_to_rank[p] = i // m
        
        sharded_params = params[rank * m : (rank + 1) * m]
        sharded_param_group = {**param_group, "params": sharded_params}
        self.optimizer.add_param_group(sharded_param_group)
