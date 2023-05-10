import torch.nn as nn
import copy

def clones(module, n):
    # 返回N层的moduleList
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])