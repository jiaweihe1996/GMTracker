import torch
import torch.nn as nn
import torch.nn.functional as F

class ReidEncoder(nn.Module):
    def __init__(self):
        super(ReidEncoder, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self._init_param()

    def _init_param(self):
        nn.init.eye_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.eye_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
       
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
