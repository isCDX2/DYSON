import torch
import torch.nn as nn


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h, bias=False)
        self.linear2 = torch.nn.Linear(num_h, num_h, bias=False)  # 2个隐层
        self.linear3 = torch.nn.Linear(num_h, num_o, bias=False)
        init_weight1 = nn.Parameter(torch.randn(num_i, num_h) * 1e-5)
        init_weight2 = nn.Parameter(torch.randn(num_h, num_h) * 1e-5)
        init_weight3 = nn.Parameter(torch.randn(num_h, num_o) * 1e-5)
        self.linear1.weight = init_weight1
        self.linear2.weight = init_weight2
        self.linear3.weight = init_weight3

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        # x = self.linear3(x)
        return x
