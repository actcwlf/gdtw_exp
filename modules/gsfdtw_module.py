from collections import defaultdict
import numbers

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit
import torch
from fsdtw import GFSDTWLayer as CppGFSDTWLayer
import torch.nn as nn
from torch.autograd import Function


def reduce_by_half(x):
    return np.array([(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)])


def get_shrink(shrunk_k , x, y, min_time_size):
    if len(x) < min_time_size or len(y) < min_time_size:
        shrunk_k.append(x)
        return
    x_shrunk = reduce_by_half(x)
    y_shrunk = reduce_by_half(y)
    get_shrink(shrunk_k, x_shrunk, y_shrunk, min_time_size)
    shrunk_k.append(x)


class GFSDTWFunction(Function):
    @staticmethod
    def forward(ctx, x, min_len, gamma, q, radius, train, *args):
        """
        注意，在fsdtw内部，使用x标记核，使用y标记输入，因此这里的x是fsdtw中的y
        :param ctx:
        :param x:
        :param params:
        :param fsdtw_ins:
        :return:
        """
        shrunk_k = [p.detach().numpy() for p in args]
        fsdtw_ins = CppGFSDTWLayer(gamma=gamma, q=q, radius=radius, input_len=min_len)
        if not train:
            fsdtw_ins.eval()
        if isinstance(x, list):
            v = torch.tensor(fsdtw_ins.forward(shrunk_k, x))
        else:
            v = torch.tensor(fsdtw_ins.forward(shrunk_k, x.detach().numpy()))
        ctx.constant = fsdtw_ins
        return v

    @staticmethod
    def backward(ctx, grad_output):
        fsdtw_ins = ctx.constant
        grad_output = grad_output.numpy()
        grads = [grad_output[i, :] for i in range(grad_output.shape[0])]
        tgx, tgy = fsdtw_ins.backward(grads)
        grad_x = None
        grad_k = [None] * len(tgx)
        if ctx.needs_input_grad[0]:
            gy = torch.tensor(tgy)
            grad_x = gy
        if ctx.needs_input_grad[-1]:
            grad_k = [torch.tensor(g).float() for g in tgx]
        return (grad_x, None, None, None, None, None, *grad_k) # 这里返回值应和forward的输入匹配


class GFSDTWLayer(nn.Module):
    def __init__(self, input_len, gamma, q, radius, **kwargs):
        super().__init__()
        self.min_len = input_len
        self.gamma = gamma
        self.q = q
        self.radius = radius
        shrunk_k = []
        get_shrink(shrunk_k, list(range(input_len)), list(range(input_len)), radius+2)
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(len(k)) / 10) for k in shrunk_k
        ])
        self.output_len = len(shrunk_k)

    def forward(self, x):
        v = GFSDTWFunction.apply(x, self.min_len, self.gamma, self.q, self.radius, self.training, *self.params)
        return v
