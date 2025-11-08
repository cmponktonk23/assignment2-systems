import torch
import torch.distributed as dist


class DDPModuleWrapper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        self.module = module
    
    def forward(self, *inputs, **kwargs):
        pass


    def finish_gradient_synchronization(self):
        pass
