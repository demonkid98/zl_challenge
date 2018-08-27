from __future__ import division, absolute_import, print_function

import torch
import torch.nn.functional as F

def gap(x):
    bs, _, h, w = x.size()
    return F.avg_pool2d(x, (h, w)).view(bs, -1)

def spp(x, f, level=2):
    bs, c, h, w = x.size()
    _xs = []
    for l in range(level):
        div = 2**l
        hstep = h // div
        wstep = w // div
        for i in range(div):
            for j in range(div):
                _xs.append(x[..., i * hstep: (i + 1) * hstep, j * wstep: (j + 1) * wstep])
    return torch.cat([gap(f(_x)) for _x in _xs], dim=-1)
