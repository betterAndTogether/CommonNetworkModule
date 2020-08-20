# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        """
        :param k: dimension size of input
        :param heads:
        """
        super().__init__()
        self.k, self.heads = k, heads

        # transform the input into three multi-head的query, key, and value matrix
        self.tokeys = nn.Linear(k, k*heads, bias=False)
        self.toquerys = nn.Linear(k, k*heads, bias=False)
        self.tovalues = nn.Linear(k, k*heads, bias=False)

        self.unifyheads = nn.Linear(k*heads, k, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toquerys(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # 这等效于对点积进行normalize
        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))
        # 矩阵相乘
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # 进行softmax归一化
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)
