import torch
import torch.nn.functional as f
from torch import nn


class Network(nn.Module):

    def __init__(self, input_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)
        self.fc3 = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = (self.fc3(x2))

        return x3
