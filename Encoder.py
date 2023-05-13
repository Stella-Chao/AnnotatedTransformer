import torch.nn as nn
from .model import clones


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()