import torch
import torch.nn as nn
from modeling.rar import RAR

class ModuleListWrapper(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        print("hello ", self.module_list)
        
    def forward(self, x, condition=[1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]):
        print(f"Input to ModuleListWrapper: {x.shape if x is not None else 'None'}")
        for module in self.module_list:
            if x is None:
                print(f"Passing through hello {module.__class__.__name__}")
            x = module(x, condition)
            if x is None:
                print(f"Output after {module.__class__.__name__}: coucou")
        return x
